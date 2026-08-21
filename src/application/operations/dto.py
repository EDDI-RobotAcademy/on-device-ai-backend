"""Operations Use Case 결과 DTO."""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from domain.operations.deployment import Deployment
from domain.operations.health import HealthReport, OnsetFinding
from domain.operations.incident import Incident
from domain.operations.inference_log import LogCoverage
from domain.operations.release_check import ReleaseCheck
from domain.operations.retraining import RetrainingDecision
from domain.operations.shadow import PromotionVerdict
from domain.operations.watch import HealthWatch


@dataclass(frozen=True, slots=True)
class DeploymentView:
    deployment_id: str
    target: str
    status: str
    current_version: int
    current_artifact: str
    model_version_id: str
    version_count: int
    rollback_count: int
    quarantine_reason: str
    history: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    versions: tuple[str, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(cls, deployment: Deployment) -> DeploymentView:
        current = deployment.current_version
        return cls(
            deployment_id=str(deployment.id),
            target=deployment.target.describe(),
            status=deployment.status.value,
            current_version=current.number,
            current_artifact=current.artifact.artifact_id,
            model_version_id=current.artifact.model_version_id,
            version_count=len(deployment.versions),
            rollback_count=deployment.rollback_count,
            quarantine_reason=deployment.quarantine_reason,
            history=deployment.history,
            versions=tuple(v.describe() for v in deployment.versions),
            _rendered=deployment.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class ReleaseCheckView:
    deployment_id: str
    verdict: str
    can_release: bool
    is_first_release: bool
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(cls, deployment_id: str, check: ReleaseCheck) -> ReleaseCheckView:
        return cls(
            deployment_id=deployment_id,
            verdict=check.verdict.value,
            can_release=check.can_release,
            is_first_release=check.is_first_release,
            findings=tuple(FindingView.of(f) for f in check.findings),
            _rendered=check.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class LogCoverageView:
    deployment_id: str
    total_count: int
    distinct_devices: int
    distinct_versions: int
    labeled_ratio: float
    digest_ratio: float
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    summary: str = ""

    @classmethod
    def of(
        cls,
        deployment_id: str,
        coverage: LogCoverage,
        findings: tuple[FindingView, ...] = (),
    ) -> LogCoverageView:
        return cls(
            deployment_id=deployment_id,
            total_count=coverage.total_count,
            distinct_devices=coverage.distinct_devices,
            distinct_versions=coverage.distinct_versions,
            labeled_ratio=coverage.labeled_ratio,
            digest_ratio=coverage.digest_ratio,
            findings=findings,
            summary=coverage.describe(),
        )

    def render(self) -> str:
        lines = [f"추론 로그 — {self.summary}"]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        else:
            lines.append("  이 로그로 답할 수 있는 질문이 다 있다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class HealthReportView:
    watch_id: str
    window_label: str
    deployment_version: int
    verdict: str
    sample_count: int
    p95_ms: float | None
    prediction_shift: float | None
    max_psi: float | None
    confidence: float | None
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    quarantine_recommended: bool = False
    quarantine_reason: str = ""
    _rendered: str = ""

    @classmethod
    def of(
        cls,
        watch_id: str,
        report: HealthReport,
        *,
        quarantine: tuple[bool, str] = (False, ""),
    ) -> HealthReportView:
        from domain.operations.health import HealthMetric

        return cls(
            watch_id=watch_id,
            window_label=report.window.label,
            deployment_version=report.deployment_version,
            verdict=report.verdict.value,
            sample_count=report.window.sample_count,
            p95_ms=report.value_of(HealthMetric.LATENCY_P95),
            prediction_shift=report.value_of(HealthMetric.PREDICTION_SHIFT),
            max_psi=report.value_of(HealthMetric.INPUT_PSI),
            confidence=report.value_of(HealthMetric.CONFIDENCE),
            findings=tuple(FindingView.of(f) for f in report.findings),
            quarantine_recommended=quarantine[0],
            quarantine_reason=quarantine[1],
            _rendered=report.render(),
        )

    def render(self) -> str:
        lines = [self._rendered]
        if self.quarantine_recommended:
            lines.append("")
            lines.append(f"  ⚠ 격리 권고: {self.quarantine_reason}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TimelineView:
    watch_id: str
    window_count: int
    table: str
    verdicts: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, watch: HealthWatch) -> TimelineView:
        timeline = watch.timeline
        return cls(
            watch_id=str(watch.id),
            window_count=len(timeline),
            table=timeline.render(),
            verdicts=tuple(r.verdict.value for r in timeline.reports),
        )

    def render(self) -> str:
        return self.table


@dataclass(frozen=True, slots=True)
class OnsetView:
    watch_id: str
    metric: str
    threshold: float
    first_exceeded: str | None
    sustained_from: str | None
    is_sustained: bool
    spike_only: bool
    _rendered: str = ""

    @classmethod
    def of(cls, watch_id: str, onset: OnsetFinding) -> OnsetView:
        return cls(
            watch_id=watch_id,
            metric=onset.metric.value,
            threshold=onset.threshold,
            first_exceeded=(
                onset.first_exceeded.window_label if onset.first_exceeded else None
            ),
            sustained_from=(
                onset.sustained_from.window_label if onset.sustained_from else None
            ),
            is_sustained=onset.is_sustained,
            spike_only=onset.spike_only,
            _rendered=onset.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class IncidentView:
    watch_id: str
    incident_id: str
    kind: str
    status: str
    window_label: str
    deployment_version: int
    summary: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    resolution: str = ""

    @classmethod
    def of(cls, watch_id: str, incident: Incident) -> IncidentView:
        return cls(
            watch_id=watch_id,
            incident_id=str(incident.incident_id),
            kind=incident.kind.value,
            status=incident.status.value,
            window_label=incident.window_label,
            deployment_version=incident.deployment_version,
            summary=incident.summary,
            findings=tuple(FindingView.of(f) for f in incident.findings),
            resolution=incident.resolution,
        )

    def render(self) -> str:
        lines = [f"{self.summary}  ({self.status})"]
        lines += [f"    {f.describe()}" for f in self.findings]
        if self.resolution:
            lines.append(f"    → {self.resolution}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ShadowView:
    deployment_id: str
    incumbent: str
    candidate: str
    sample_count: int
    agreement_ratio: float
    labeled_count: int
    incumbent_accuracy: float | None
    candidate_accuracy: float | None
    accuracy_gain: float | None
    verdict: str
    promote: bool
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(cls, deployment_id: str, verdict: PromotionVerdict) -> ShadowView:
        run = verdict.run
        return cls(
            deployment_id=deployment_id,
            incumbent=run.incumbent_label,
            candidate=run.candidate_label,
            sample_count=run.sample_count,
            agreement_ratio=run.agreement_ratio,
            labeled_count=run.labeled_count,
            incumbent_accuracy=run.incumbent_accuracy,
            candidate_accuracy=run.candidate_accuracy,
            accuracy_gain=run.accuracy_gain,
            verdict=verdict.verdict.value,
            promote=verdict.promote,
            findings=tuple(FindingView.of(f) for f in verdict.findings),
            _rendered=verdict.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class RetrainingView:
    watch_id: str
    needed: bool
    can_start: bool
    urgency: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)
    findings: tuple[FindingView, ...] = field(default_factory=tuple)
    _rendered: str = ""

    @classmethod
    def of(cls, watch_id: str, decision: RetrainingDecision) -> RetrainingView:
        return cls(
            watch_id=watch_id,
            needed=decision.needed,
            can_start=decision.can_start,
            urgency=decision.urgency.value,
            reasons=tuple(reason.value for reason in decision.reasons),
            blockers=decision.blockers,
            findings=tuple(FindingView.of(f) for f in decision.findings),
            _rendered=decision.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class WatchView:
    watch_id: str
    deployment_id: str
    window_count: int
    incident_count: int
    open_incident_count: int
    baseline_p95_ms: float
    _rendered: str = ""

    @classmethod
    def of(cls, watch: HealthWatch) -> WatchView:
        return cls(
            watch_id=str(watch.id),
            deployment_id=str(watch.deployment_id),
            window_count=len(watch.reports),
            incident_count=len(watch.incidents),
            open_incident_count=len(watch.open_incidents),
            baseline_p95_ms=watch.baseline_p95_ms,
            _rendered=watch.render(),
        )

    def render(self) -> str:
        return self._rendered
