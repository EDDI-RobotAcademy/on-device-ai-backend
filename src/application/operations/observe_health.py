"""AI의 판단을 받아 적고, 창 단위로 지켜본다. (실습 5-3 ~ 5-7)

세 측정기를 부르고 세 Policy 에 건넨다. 판단은 하나도 하지 않는다.

    LatencyMeasurer      → LatencyPolicy         느려졌는가        (5-5)
    PredictionMixMeasurer → PredictionDriftPolicy 답이 변했는가     (5-6)
    InputDriftMeasurer   → DriftPolicy           입력이 변했는가   (5-7)

셋을 **한 창에서 함께** 본다. 따로 보면 원인을 못 짚는다.
    입력이 변했는데 예측이 그대로다  → 아직 견디고 있다
    입력도 예측도 변했다             → 세상이 바뀐 것이다
    입력은 그대로인데 예측이 변했다  → 모델이나 전처리를 의심한다
    지연시간만 변했다                → 모델이 아니라 환경이다
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.operations.dto import HealthReportView, LogCoverageView
from application.operations.support import commit, load_deployment, watch_for
from application.shared.ports import EventPublisher
from domain.operations.drift import DriftPolicy
from domain.operations.health import HealthReport
from domain.operations.identifiers import IncidentId
from domain.operations.incident import IncidentPolicy
from domain.operations.inference_log import (
    InferenceLogPolicy,
    InferenceRecord,
    summarize,
)
from domain.operations.latency import LatencyPolicy
from domain.operations.ports import (
    DeploymentRepository,
    HealthWatchRepository,
    InferenceLogStore,
    InputDriftMeasurer,
    LatencyMeasurer,
    PredictionMixMeasurer,
)
from domain.operations.prediction_mix import PredictionDriftPolicy
from domain.operations.window import ObservationWindow, WindowPolicy
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class IngestInferenceLogCommand:
    """디바이스가 올린 판단 기록을 받아 적는다. (실습 5-3)"""

    deployment_id: str
    records: tuple[InferenceRecord, ...]
    policy: InferenceLogPolicy = field(default_factory=InferenceLogPolicy)


class IngestInferenceLog:
    def __init__(
        self, deployments: DeploymentRepository, logs: InferenceLogStore
    ) -> None:
        self._deployments = deployments
        self._logs = logs

    def execute(self, command: IngestInferenceLogCommand) -> LogCoverageView:
        deployment = load_deployment(self._deployments, command.deployment_id)
        self._logs.append(command.records)

        coverage = summarize(command.records)
        findings = command.policy.inspect(coverage)
        from application.data.dto import FindingView

        return LogCoverageView.of(
            str(deployment.id),
            coverage,
            tuple(FindingView.of(f) for f in findings),
        )


@dataclass(frozen=True, slots=True)
class ObserveHealthCommand:
    """창 하나를 본다. (실습 5-4 ~ 5-7)"""

    deployment_id: str
    window: ObservationWindow
    latency_policy: LatencyPolicy
    mix_policy: PredictionDriftPolicy = field(default_factory=PredictionDriftPolicy)
    drift_policy: DriftPolicy = field(default_factory=DriftPolicy)
    log_policy: InferenceLogPolicy = field(default_factory=InferenceLogPolicy)
    window_policy: WindowPolicy = field(default_factory=WindowPolicy)
    incident_policy: IncidentPolicy = field(default_factory=IncidentPolicy)
    open_incident: bool = True
    measure_drift: bool = True


class ObserveHealth:
    def __init__(
        self,
        deployments: DeploymentRepository,
        watches: HealthWatchRepository,
        logs: InferenceLogStore,
        latency: LatencyMeasurer,
        mix: PredictionMixMeasurer,
        drift: InputDriftMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._deployments = deployments
        self._watches = watches
        self._logs = logs
        self._latency = latency
        self._mix = mix
        self._drift = drift
        self._publisher = publisher

    def execute(self, command: ObserveHealthCommand) -> HealthReportView:
        deployment = load_deployment(self._deployments, command.deployment_id)
        watch = watch_for(self._watches, deployment.id)

        records = self._logs.records_in(deployment.id, command.window)
        window = _resize(command.window, len(records))

        findings: list[Finding] = []
        for message in command.window_policy.inspect(window):
            findings.append(
                Finding(
                    code="OPS_WINDOW_TOO_SMALL",
                    message=message,
                    severity=Severity.WARNING,
                    subject=window.label,
                    measured=float(window.sample_count),
                    threshold=float(command.window_policy.min_sample_count),
                )
            )

        coverage = summarize(records)
        findings.extend(command.log_policy.inspect(coverage))

        latency = mix = drift = None
        if records:
            latency = self._latency.measure(deployment.id, window)
            findings.extend(
                command.latency_policy.inspect(latency, watch.baseline_p95_ms)
            )

            mix = self._mix.measure(deployment.id, window)
            findings.extend(command.mix_policy.inspect(mix, watch.baseline_mix))

            if command.measure_drift:
                drift = self._drift.measure(deployment.id, window)
                findings.extend(command.drift_policy.inspect(drift))

        report = HealthReport(
            window=window,
            deployment_version=_version_of(records, deployment),
            findings=tuple(findings),
            latency=latency,
            mix=mix,
            drift=drift,
            baseline_mix=watch.baseline_mix,
        )
        watch.record(report)
        if command.open_incident:
            watch.open_incident(
                IncidentId.of(f"inc-{watch.id}-{window.label}"),
                report,
                command.incident_policy,
            )
        commit(self._watches, watch, self._publisher)

        return HealthReportView.of(
            str(watch.id),
            report,
            quarantine=command.incident_policy.should_quarantine(report),
        )


def _resize(window: ObservationWindow, sample_count: int) -> ObservationWindow:
    """창의 표본 수는 **실제로 들어온 로그 수**다.

    창을 만들 때 미리 적어 둔 숫자를 믿으면, 로그가 안 올라온 구간을 못 알아챈다.
    """
    from dataclasses import replace

    return replace(window, sample_count=sample_count)


def _version_of(records, deployment) -> int:  # noqa: ANN001
    """이 창에서 돌고 있던 배포 버전."""
    versions = {r.deployment_version for r in records}
    if len(versions) == 1:
        return versions.pop()
    return deployment.current_version.number


@dataclass(frozen=True, slots=True)
class RebaselineCommand:
    """현장 안정 구간을 새 기준으로 잡는다. (실습 5-6)"""

    deployment_id: str
    window: ObservationWindow
    reason: str


class RebaselineWatch:
    def __init__(
        self,
        deployments: DeploymentRepository,
        watches: HealthWatchRepository,
        mix: PredictionMixMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._deployments = deployments
        self._watches = watches
        self._mix = mix
        self._publisher = publisher

    def execute(self, command: RebaselineCommand) -> dict[str, float]:
        deployment = load_deployment(self._deployments, command.deployment_id)
        watch = watch_for(self._watches, deployment.id)

        measured = self._mix.measure(deployment.id, command.window)
        watch.rebaseline(measured.shares, command.reason)
        commit(self._watches, watch, self._publisher)
        return watch.baseline_mix
