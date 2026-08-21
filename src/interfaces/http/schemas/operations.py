"""Operations API 의 요청/응답 DTO 와 Domain 매퍼. (모듈 5)

이 Context 의 DTO 에는 특징이 하나 있다.
**요청에 시각(`released_at`, `occurred_at`)을 넣을 수 있다.**

운영 기록은 데이터의 시각을 따라야 할 때가 있다 —
지난 로그를 다시 넣거나(backfill), 실습에서 4일치를 몇 초에 재생할 때가 그렇다.
"지금"이 언제나 맞는 답은 아니다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from application.operations.dto import (
    DeploymentView,
    HealthReportView,
    IncidentView,
    LogCoverageView,
    OnsetView,
    ReleaseCheckView,
    RetrainingView,
    ShadowView,
    TimelineView,
    WatchView,
)
from domain.operations.drift import DriftPolicy
from domain.operations.health import HealthMetric
from domain.operations.incident import IncidentPolicy
from domain.operations.inference_log import InferenceLogPolicy, InferenceRecord
from domain.operations.latency import LatencyPolicy
from domain.operations.prediction_mix import PredictionDriftPolicy
from domain.operations.release_check import ReleasePolicy
from domain.operations.retraining import LabelSupply, RetrainingPolicy
from domain.operations.shadow import PromotionPolicy
from domain.operations.target import DeploymentTarget, TargetKind
from domain.operations.window import ObservationWindow, WindowPolicy
from interfaces.http.schemas.dataset import FindingResponse


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
class ObservationWindowRequest(BaseModel):
    label: str = Field(examples=["05-23 00시"])
    started_at: str = Field(examples=["2026-05-23 00:00:00"])
    ended_at: str = Field(examples=["2026-05-23 07:59:59"])
    device_id: str | None = None
    """한 대만 볼 때 적는다. 전체를 보면 비운다. (실습 5-5)"""

    def to_domain(self) -> ObservationWindow:
        return ObservationWindow(
            label=self.label,
            started_at=self.started_at,
            ended_at=self.ended_at,
            sample_count=0,  # 실제 표본 수는 로그에서 세어 채운다
            device_id=self.device_id,
        )


class DeploymentTargetRequest(BaseModel):
    kind: Literal["DEVICE", "DEVICE_GROUP", "FLEET"] = "DEVICE_GROUP"
    identifier: str = Field(examples=["LINE-3"])
    name: str = ""
    device_count: int = Field(default=1, ge=1)

    def to_domain(self) -> DeploymentTarget:
        return DeploymentTarget(
            kind=TargetKind(self.kind),
            identifier=self.identifier,
            name=self.name,
            device_count=self.device_count,
        )


# ---------------------------------------------------------------------------
# 정책
# ---------------------------------------------------------------------------
class ReleasePolicyRequest(BaseModel):
    require_selected: bool = True
    require_preprocessing: bool = True
    require_baseline: bool = True
    max_first_release_devices: int = 10

    def to_domain(self) -> ReleasePolicy:
        return ReleasePolicy(**self.model_dump())


class LatencyPolicyRequest(BaseModel):
    cycle_budget_ms: float = Field(gt=0, examples=[30.0])
    max_regression_ratio: float = 3.0
    max_jitter_ratio: float = 3.0
    max_timeout_ratio: float = 0.001

    def to_domain(self) -> LatencyPolicy:
        return LatencyPolicy(**self.model_dump())


class PredictionDriftPolicyRequest(BaseModel):
    max_shift: float = Field(default=0.15, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    max_confidence_drop: float = Field(default=0.15, ge=0.0, le=1.0)
    surge_factor: float = Field(default=3.0, gt=0)
    critical_labels: list[str] = Field(default_factory=list)

    def to_domain(self) -> PredictionDriftPolicy:
        body = self.model_dump()
        body["critical_labels"] = frozenset(body["critical_labels"])
        return PredictionDriftPolicy(**body)


class DriftPolicyRequest(BaseModel):
    max_psi: float = 0.2
    watch_psi: float = 0.1
    max_out_of_range_ratio: float = 0.01
    max_drifted_field_count: int = 2

    def to_domain(self) -> DriftPolicy:
        return DriftPolicy(**self.model_dump())


class InferenceLogPolicyRequest(BaseModel):
    min_digest_ratio: float = 0.9
    min_labeled_ratio: float = 0.0

    def to_domain(self) -> InferenceLogPolicy:
        return InferenceLogPolicy(**self.model_dump())


class WindowPolicyRequest(BaseModel):
    min_sample_count: int = 100

    def to_domain(self) -> WindowPolicy:
        return WindowPolicy(**self.model_dump())


class IncidentPolicyRequest(BaseModel):
    quarantine_on_critical: bool = True
    quarantine_on_warning_count: int = 4

    def to_domain(self) -> IncidentPolicy:
        return IncidentPolicy(**self.model_dump())


class PromotionPolicyRequest(BaseModel):
    min_sample_count: int = 500
    min_labeled_count: int = 100
    min_accuracy_gain: float = 0.0
    max_slowdown_ratio: float = 1.2
    min_agreement_ratio: float = 0.0
    require_labels: bool = True

    def to_domain(self) -> PromotionPolicy:
        return PromotionPolicy(**self.model_dump())


class RetrainingPolicyRequest(BaseModel):
    drift_psi_threshold: float = 0.2
    prediction_shift_threshold: float = 0.15
    sustained_windows: int = Field(default=3, ge=1)
    min_new_labels: int = 500
    min_labels_per_class: int = 30
    confidence_floor: float = 0.6
    measured_accuracy_floor: float | None = None

    def to_domain(self) -> RetrainingPolicy:
        return RetrainingPolicy(**self.model_dump())


class LabelSupplyRequest(BaseModel):
    total_records: int = Field(ge=0)
    labeled_records: int = Field(ge=0)
    labeled_since_deploy: int = 0
    minority_label_counts: dict[str, int] = Field(default_factory=dict)

    def to_domain(self) -> LabelSupply:
        return LabelSupply(**self.model_dump())


# ---------------------------------------------------------------------------
# 요청
# ---------------------------------------------------------------------------
class DeployModelRequest(BaseModel):
    deployment_id: str = Field(examples=["dep-line3"])
    optimization_run_id: str = Field(examples=["opt-power"])
    target: DeploymentTargetRequest
    training_run_id: str | None = Field(default=None, examples=["run-opt"])
    artifact_label: str | None = None
    watch_id: str | None = None
    note: str = ""
    released_at: str | None = None
    require_selected: bool = True
    policy: ReleasePolicyRequest = Field(default_factory=ReleasePolicyRequest)


class ReleaseVersionRequest(BaseModel):
    optimization_run_id: str
    training_run_id: str | None = None
    artifact_label: str | None = None
    note: str = ""
    released_at: str | None = None
    require_selected: bool = True
    policy: ReleasePolicyRequest = Field(default_factory=ReleasePolicyRequest)


class InferenceRecordRequest(BaseModel):
    """디바이스가 올리는 판단 기록 하나. (실습 5-3)"""

    occurred_at: str
    device_id: str
    deployment_version: int = Field(ge=1)
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    input_digest: str = ""
    ground_truth: str | None = None

    def to_domain(self) -> InferenceRecord:
        return InferenceRecord(**self.model_dump())


class IngestInferenceLogRequest(BaseModel):
    records: list[InferenceRecordRequest]
    policy: InferenceLogPolicyRequest = Field(
        default_factory=InferenceLogPolicyRequest
    )


class ObserveHealthRequest(BaseModel):
    window: ObservationWindowRequest
    latency_policy: LatencyPolicyRequest
    mix_policy: PredictionDriftPolicyRequest = Field(
        default_factory=PredictionDriftPolicyRequest
    )
    drift_policy: DriftPolicyRequest = Field(default_factory=DriftPolicyRequest)
    log_policy: InferenceLogPolicyRequest = Field(
        default_factory=InferenceLogPolicyRequest
    )
    window_policy: WindowPolicyRequest = Field(default_factory=WindowPolicyRequest)
    incident_policy: IncidentPolicyRequest = Field(
        default_factory=IncidentPolicyRequest
    )
    open_incident: bool = True
    measure_drift: bool = True


class RebaselineRequest(BaseModel):
    window: ObservationWindowRequest
    reason: str = Field(min_length=1, examples=["배포 직후 안정 구간"])


class FindOnsetRequest(BaseModel):
    metric: Literal[
        "LATENCY_P95", "PREDICTION_SHIFT", "INPUT_PSI", "CONFIDENCE"
    ] = "INPUT_PSI"
    threshold: float = 0.2
    consecutive: int = Field(default=3, ge=1)

    def metric_target(self) -> HealthMetric:
        return HealthMetric(self.metric)


class QuarantineRequest(BaseModel):
    reason: str = ""
    """비우면 최근 관측의 CRITICAL 소견을 이유로 쓴다."""

    occurred_at: str | None = None
    policy: IncidentPolicyRequest = Field(default_factory=IncidentPolicyRequest)


class ResumeRequest(BaseModel):
    reason: str = Field(min_length=1)
    occurred_at: str | None = None


class RollbackRequest(BaseModel):
    to_version: int = Field(ge=1)
    reason: str = Field(min_length=1)
    occurred_at: str | None = None


class ResolveIncidentRequest(BaseModel):
    resolution: str = Field(min_length=1)


class CompareShadowRequest(BaseModel):
    window: ObservationWindowRequest
    candidate_artifact_id: str
    policy: PromotionPolicyRequest = Field(default_factory=PromotionPolicyRequest)


class DecideRetrainingRequest(BaseModel):
    supply: LabelSupplyRequest | None = None
    measured_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    policy: RetrainingPolicyRequest = Field(default_factory=RetrainingPolicyRequest)


# ---------------------------------------------------------------------------
# 응답
# ---------------------------------------------------------------------------
class DeploymentResponse(BaseModel):
    deployment_id: str
    target: str
    status: str
    current_version: int
    current_artifact: str
    model_version_id: str
    version_count: int
    rollback_count: int
    quarantine_reason: str = ""
    versions: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: DeploymentView) -> DeploymentResponse:
        return cls(
            deployment_id=view.deployment_id,
            target=view.target,
            status=view.status,
            current_version=view.current_version,
            current_artifact=view.current_artifact,
            model_version_id=view.model_version_id,
            version_count=view.version_count,
            rollback_count=view.rollback_count,
            quarantine_reason=view.quarantine_reason,
            versions=list(view.versions),
            history=[{"at": at, "what": what} for at, what in view.history],
            report=view.render(),
        )


class ReleaseCheckResponse(BaseModel):
    deployment_id: str
    verdict: str
    can_release: bool
    is_first_release: bool
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: ReleaseCheckView) -> ReleaseCheckResponse:
        return cls(
            deployment_id=view.deployment_id,
            verdict=view.verdict,
            can_release=view.can_release,
            is_first_release=view.is_first_release,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class DeployResponse(BaseModel):
    deployment: DeploymentResponse
    check: ReleaseCheckResponse
    watch_id: str = ""


class LogCoverageResponse(BaseModel):
    deployment_id: str
    total_count: int
    distinct_devices: int
    distinct_versions: int
    labeled_ratio: float
    digest_ratio: float
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: LogCoverageView) -> LogCoverageResponse:
        return cls(
            deployment_id=view.deployment_id,
            total_count=view.total_count,
            distinct_devices=view.distinct_devices,
            distinct_versions=view.distinct_versions,
            labeled_ratio=view.labeled_ratio,
            digest_ratio=view.digest_ratio,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class HealthReportResponse(BaseModel):
    watch_id: str
    window_label: str
    deployment_version: int
    verdict: str
    sample_count: int
    p95_ms: float | None = None
    prediction_shift: float | None = None
    max_psi: float | None = None
    confidence: float | None = None
    quarantine_recommended: bool = False
    quarantine_reason: str = ""
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: HealthReportView) -> HealthReportResponse:
        return cls(
            watch_id=view.watch_id,
            window_label=view.window_label,
            deployment_version=view.deployment_version,
            verdict=view.verdict,
            sample_count=view.sample_count,
            p95_ms=view.p95_ms,
            prediction_shift=view.prediction_shift,
            max_psi=view.max_psi,
            confidence=view.confidence,
            quarantine_recommended=view.quarantine_recommended,
            quarantine_reason=view.quarantine_reason,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class TimelineResponse(BaseModel):
    watch_id: str
    window_count: int
    verdicts: list[str] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: TimelineView) -> TimelineResponse:
        return cls(
            watch_id=view.watch_id,
            window_count=view.window_count,
            verdicts=list(view.verdicts),
            report=view.render(),
        )


class OnsetResponse(BaseModel):
    watch_id: str
    metric: str
    threshold: float
    first_exceeded: str | None = None
    sustained_from: str | None = None
    is_sustained: bool
    spike_only: bool
    report: str

    @classmethod
    def from_view(cls, view: OnsetView) -> OnsetResponse:
        return cls(
            watch_id=view.watch_id,
            metric=view.metric,
            threshold=view.threshold,
            first_exceeded=view.first_exceeded,
            sustained_from=view.sustained_from,
            is_sustained=view.is_sustained,
            spike_only=view.spike_only,
            report=view.render(),
        )


class IncidentResponse(BaseModel):
    watch_id: str
    incident_id: str
    kind: str
    status: str
    window_label: str
    deployment_version: int
    summary: str
    resolution: str = ""
    findings: list[FindingResponse] = Field(default_factory=list)

    @classmethod
    def from_view(cls, view: IncidentView) -> IncidentResponse:
        return cls(
            watch_id=view.watch_id,
            incident_id=view.incident_id,
            kind=view.kind,
            status=view.status,
            window_label=view.window_label,
            deployment_version=view.deployment_version,
            summary=view.summary,
            resolution=view.resolution,
            findings=[FindingResponse.from_view(f) for f in view.findings],
        )


class WatchResponse(BaseModel):
    watch_id: str
    deployment_id: str
    window_count: int
    incident_count: int
    open_incident_count: int
    baseline_p95_ms: float
    report: str

    @classmethod
    def from_view(cls, view: WatchView) -> WatchResponse:
        return cls(
            watch_id=view.watch_id,
            deployment_id=view.deployment_id,
            window_count=view.window_count,
            incident_count=view.incident_count,
            open_incident_count=view.open_incident_count,
            baseline_p95_ms=view.baseline_p95_ms,
            report=view.render(),
        )


class ShadowResponse(BaseModel):
    deployment_id: str
    incumbent: str
    candidate: str
    sample_count: int
    agreement_ratio: float
    labeled_count: int
    incumbent_accuracy: float | None = None
    candidate_accuracy: float | None = None
    accuracy_gain: float | None = None
    verdict: str
    promote: bool
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: ShadowView) -> ShadowResponse:
        return cls(
            deployment_id=view.deployment_id,
            incumbent=view.incumbent,
            candidate=view.candidate,
            sample_count=view.sample_count,
            agreement_ratio=view.agreement_ratio,
            labeled_count=view.labeled_count,
            incumbent_accuracy=view.incumbent_accuracy,
            candidate_accuracy=view.candidate_accuracy,
            accuracy_gain=view.accuracy_gain,
            verdict=view.verdict,
            promote=view.promote,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class RetrainingResponse(BaseModel):
    watch_id: str
    needed: bool
    can_start: bool
    urgency: str
    reasons: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: RetrainingView) -> RetrainingResponse:
        return cls(
            watch_id=view.watch_id,
            needed=view.needed,
            can_start=view.can_start,
            urgency=view.urgency,
            reasons=list(view.reasons),
            blockers=list(view.blockers),
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class BaselineResponse(BaseModel):
    deployment_id: str
    baseline_mix: dict[str, float]
