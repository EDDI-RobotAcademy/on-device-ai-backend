"""Fleet API 의 요청/응답 DTO 와 Domain 매퍼. (모듈 6)

**이 파일에도 AWS 가 없다.** HTTP 를 쓰는 쪽은 데이터가 S3 로 가는지
GCS 로 가는지 알 필요가 없다 — 알면 나중에 바꿀 때 클라이언트까지 고쳐야 한다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from application.fleet.dto import (
    DatasetBuildView,
    FleetView,
    LakeLayoutView,
    LineageView,
    ReleaseView,
    RolloutView,
    TrainingJobView,
    UplinkView,
    WaveView,
)
from domain.fleet.dataset_build import DatasetBuildPolicy, SourceWindow
from domain.fleet.device import Device, DeviceStatus, FleetHealthPolicy
from domain.fleet.object_key import KeyLayout, KeyLayoutPolicy
from domain.fleet.release import ReleaseBundle, ReleaseChannel, ReleasePolicy
from domain.fleet.rollout import RolloutPolicy
from domain.fleet.training_job import ComputeSpec, TrainingBudgetPolicy
from domain.fleet.uplink import UplinkBatch, UplinkKind, UplinkPolicy
from interfaces.http.schemas.dataset import FindingResponse

StatusName = Literal[
    "HEALTHY", "STALE", "UNREACHABLE", "QUARANTINED", "RETIRED"
]
ChannelName = Literal["CANARY", "STABLE", "ARCHIVED"]
UplinkKindName = Literal["INFERENCE_LOG", "RAW_SAMPLE", "HEALTH_REPORT", "INCIDENT"]


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
class DeviceRequest(BaseModel):
    device_id: str = Field(examples=["DEV-00"])
    group: str = Field(examples=["pilot"])
    current_version: str = ""
    last_seen_at: str = ""
    site: str = ""
    note: str = ""

    def to_domain(self) -> Device:
        return Device(**self.model_dump())


class UplinkBatchRequest(BaseModel):
    device_id: str
    kind: UplinkKindName = "INFERENCE_LOG"
    window_start: str
    window_end: str
    record_count: int = Field(ge=0)
    payload_bytes: int = Field(ge=0)
    checksum: str = ""
    fields: list[str] = Field(default_factory=list)

    def to_domain(self) -> UplinkBatch:
        body = self.model_dump()
        body["kind"] = UplinkKind(body["kind"])
        body["fields"] = tuple(body["fields"])
        return UplinkBatch(**body)


class ReleaseBundleRequest(BaseModel):
    release_id: str
    version: str = Field(examples=["v2.0.0"])
    model_version_id: str
    artifact_uri: str
    artifact_bytes: int = Field(gt=0)
    checksum: str = Field(min_length=1)
    runtime: str = Field(examples=["TFLITE"])
    precision: str = Field(examples=["FP16"])
    class_labels: list[str]
    input_fields: list[str] = Field(default_factory=list)
    normalization: dict[str, list[float]] = Field(default_factory=dict)
    expected_p95_ms: float = 0.0
    expected_class_mix: dict[str, float] = Field(default_factory=dict)
    sample_interval_seconds: int = 0
    window_length: int = 0
    channel: ChannelName = "CANARY"
    built_at: str = ""
    source_build_id: str = ""
    source_job_id: str = ""

    def to_domain(self) -> ReleaseBundle:
        body = self.model_dump()
        body["class_labels"] = tuple(body["class_labels"])
        body["input_fields"] = tuple(body["input_fields"])
        body["normalization"] = {
            name: (float(stats[0]), float(stats[1]))
            for name, stats in body["normalization"].items()
            if len(stats) == 2
        }
        body["channel"] = ReleaseChannel(body["channel"])
        return ReleaseBundle(**body)


class ComputeSpecRequest(BaseModel):
    instance_type: str = Field(examples=["ml.m5.large"])
    instance_count: int = Field(default=1, ge=1)
    max_runtime_seconds: int = Field(default=3_600, ge=60)
    hourly_cost_usd: float = Field(default=0.0, ge=0.0)

    def to_domain(self) -> ComputeSpec:
        return ComputeSpec(**self.model_dump())


# ---------------------------------------------------------------------------
# 정책
# ---------------------------------------------------------------------------
class UplinkPolicyRequest(BaseModel):
    max_batch_kib: float = 512.0
    daily_budget_kib_per_device: float = 20_480.0
    forbidden_fields: list[str] = Field(
        default_factory=lambda: [
            "operator_name",
            "employee_id",
            "badge_id",
            "phone",
            "email",
        ]
    )
    min_records_per_batch: int = 10
    require_checksum: bool = True

    def to_domain(self) -> UplinkPolicy:
        body = self.model_dump()
        body["forbidden_fields"] = frozenset(body["forbidden_fields"])
        return UplinkPolicy(**body)


class KeyLayoutRequest(BaseModel):
    prefix: str = "uplinks"
    order: list[str] = Field(default_factory=lambda: ["kind", "device", "date", "hour"])

    def to_domain(self) -> KeyLayout:
        return KeyLayout(prefix=self.prefix, order=tuple(self.order))


class KeyLayoutPolicyRequest(BaseModel):
    min_mean_object_kib: float = 64.0
    max_objects_per_prefix: int = 10_000
    required_partitions: list[str] = Field(
        default_factory=lambda: ["device", "date"]
    )

    def to_domain(self) -> KeyLayoutPolicy:
        body = self.model_dump()
        body["required_partitions"] = tuple(body["required_partitions"])
        return KeyLayoutPolicy(**body)


class FleetHealthPolicyRequest(BaseModel):
    max_stale_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    max_version_count: int = 3
    min_dominant_share: float = Field(default=0.8, ge=0.0, le=1.0)
    max_never_reported: int = 0

    def to_domain(self) -> FleetHealthPolicy:
        return FleetHealthPolicy(**self.model_dump())


class DatasetBuildPolicyRequest(BaseModel):
    min_records: int = 5_000
    min_labeled: int = 500
    min_labels_per_class: int = 30
    max_device_share: float = Field(default=0.5, ge=0.0, le=1.0)
    min_devices: int = 2
    require_exclusion_reasons: bool = True

    def to_domain(self) -> DatasetBuildPolicy:
        return DatasetBuildPolicy(**self.model_dump())


class TrainingBudgetPolicyRequest(BaseModel):
    max_cost_usd: float = 20.0
    max_runtime_seconds: int = 7_200
    require_cost_estimate: bool = True
    min_metrics: dict[str, float] = Field(default_factory=dict)

    def to_domain(self) -> TrainingBudgetPolicy:
        return TrainingBudgetPolicy(**self.model_dump())


class ReleasePolicyRequest(BaseModel):
    max_artifact_kib: float = 256.0
    max_fleet_transfer_mib: float = 512.0
    require_preprocessing: bool = True
    require_baseline: bool = True
    require_lineage: bool = True

    def to_domain(self) -> ReleasePolicy:
        return ReleasePolicy(**self.model_dump())


class RolloutPolicyRequest(BaseModel):
    max_failure_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    min_reported_before_advance: float = Field(default=0.7, ge=0.0, le=1.0)
    max_unreachable_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    require_canary_first: bool = True
    min_canary_devices: int = 2

    def to_domain(self) -> RolloutPolicy:
        return RolloutPolicy(**self.model_dump())


# ---------------------------------------------------------------------------
# 요청
# ---------------------------------------------------------------------------
class CreateFleetRequest(BaseModel):
    fleet_id: str = Field(examples=["line3"])
    name: str = Field(examples=["3라인 전력 감시"])
    devices: list[DeviceRequest] = Field(default_factory=list)


class IngestUplinkRequest(BaseModel):
    batch: UplinkBatchRequest
    body_base64: str = Field(default="", description="올린 내용 (base64)")
    part: int = 0
    layout: KeyLayoutRequest = Field(default_factory=KeyLayoutRequest)
    policy: UplinkPolicyRequest = Field(default_factory=UplinkPolicyRequest)


class InspectLakeRequest(BaseModel):
    layout: KeyLayoutRequest = Field(default_factory=KeyLayoutRequest)
    policy: KeyLayoutPolicyRequest = Field(default_factory=KeyLayoutPolicyRequest)
    filters: dict[str, str] = Field(default_factory=dict)


class MarkDeviceRequest(BaseModel):
    status: StatusName
    note: str = ""

    def status_target(self) -> DeviceStatus:
        return DeviceStatus(self.status)


class SweepRequest(BaseModel):
    now: str
    stale_after: str
    unreachable_after: str


class SourceWindowRequest(BaseModel):
    started_at: str
    ended_at: str
    reason: str = ""

    def to_domain(self) -> SourceWindow:
        return SourceWindow(**self.model_dump())


class BuildDatasetRequest(BaseModel):
    build_id: str
    window: SourceWindowRequest
    record_counts: dict[str, int]
    labeled_counts: dict[str, int]
    label_distribution: dict[str, int]
    include_devices: list[str] = Field(default_factory=list)
    policy: DatasetBuildPolicyRequest = Field(
        default_factory=DatasetBuildPolicyRequest
    )


class SubmitTrainingRequest(BaseModel):
    job_id: str
    dataset_uri: str
    output_uri: str
    compute: ComputeSpecRequest
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    policy: TrainingBudgetPolicyRequest = Field(
        default_factory=TrainingBudgetPolicyRequest
    )


class PublishReleaseRequest(BaseModel):
    bundle: ReleaseBundleRequest
    policy: ReleasePolicyRequest = Field(default_factory=ReleasePolicyRequest)


class PromoteReleaseRequest(BaseModel):
    version: str
    channel: ChannelName = "CANARY"

    def channel_target(self) -> ReleaseChannel:
        return ReleaseChannel(self.channel)


class PlanRolloutRequest(BaseModel):
    rollout_id: str
    version: str
    wave_sizes: list[int] = Field(default_factory=lambda: [2, 8, 9999])
    group_order: list[str] = Field(default_factory=list)
    policy: RolloutPolicyRequest = Field(default_factory=RolloutPolicyRequest)
    occurred_at: str | None = None


class CollectWaveRequest(BaseModel):
    policy: RolloutPolicyRequest = Field(default_factory=RolloutPolicyRequest)
    occurred_at: str | None = None


class AdvanceRolloutRequest(BaseModel):
    fleet_id: str
    policy: RolloutPolicyRequest = Field(default_factory=RolloutPolicyRequest)
    occurred_at: str | None = None


class HaltRolloutRequest(BaseModel):
    reason: str = Field(min_length=1)
    occurred_at: str | None = None


class RollbackRolloutRequest(BaseModel):
    fleet_id: str
    new_rollout_id: str
    reason: str = Field(min_length=1)
    to_version: str = ""
    wave_sizes: list[int] = Field(default_factory=lambda: [2, 8, 9999])
    policy: RolloutPolicyRequest = Field(default_factory=RolloutPolicyRequest)
    occurred_at: str | None = None


class ApplyOutcomesRequest(BaseModel):
    fleet_id: str
    seen_at: str


class TraceLineageRequest(BaseModel):
    source_devices: list[str] = Field(default_factory=list)
    window: str = ""


# ---------------------------------------------------------------------------
# 응답
# ---------------------------------------------------------------------------
class FleetResponse(BaseModel):
    fleet_id: str
    name: str
    size: int
    channels: str
    reachable: int
    stale_ratio: float
    version_count: int
    dominant_version: str
    dominant_share: float
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: FleetView) -> FleetResponse:
        return cls(
            fleet_id=view.fleet_id,
            name=view.name,
            size=view.size,
            channels=view.channels,
            reachable=view.reachable,
            stale_ratio=view.stale_ratio,
            version_count=view.version_count,
            dominant_version=view.dominant_version,
            dominant_share=view.dominant_share,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class UplinkResponse(BaseModel):
    fleet_id: str
    device_id: str
    accepted: bool
    uri: str
    record_count: int
    payload_kib: float
    sent_today_kib: float
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: UplinkView) -> UplinkResponse:
        return cls(
            fleet_id=view.fleet_id,
            device_id=view.device_id,
            accepted=view.accepted,
            uri=view.uri,
            record_count=view.record_count,
            payload_kib=view.payload_kib,
            sent_today_kib=view.sent_today_kib,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class LakeLayoutResponse(BaseModel):
    fleet_id: str
    prefix: str
    object_count: int
    total_mib: float
    mean_kib: float
    distinct_prefixes: int
    can_narrow: bool
    narrowed_prefix: str
    narrowed_count: int
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: LakeLayoutView) -> LakeLayoutResponse:
        return cls(
            fleet_id=view.fleet_id,
            prefix=view.prefix,
            object_count=view.object_count,
            total_mib=view.total_mib,
            mean_kib=view.mean_kib,
            distinct_prefixes=view.distinct_prefixes,
            can_narrow=view.can_narrow,
            narrowed_prefix=view.narrowed_prefix,
            narrowed_count=view.narrowed_count,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class DatasetBuildResponse(BaseModel):
    fleet_id: str
    build_id: str
    verdict: str
    can_build: bool
    dataset_uri: str
    total_records: int
    total_labeled: int
    device_count: int
    excluded: list[dict[str, str]] = Field(default_factory=list)
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: DatasetBuildView) -> DatasetBuildResponse:
        return cls(
            fleet_id=view.fleet_id,
            build_id=view.build_id,
            verdict=view.verdict,
            can_build=view.can_build,
            dataset_uri=view.dataset_uri,
            total_records=view.total_records,
            total_labeled=view.total_labeled,
            device_count=view.device_count,
            excluded=[
                {"device_id": device, "reason": reason}
                for device, reason in view.excluded
            ],
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    is_terminal: bool
    succeeded: bool
    dataset_uri: str
    artifact_uri: str
    failure_reason: str
    instance: str
    worst_case_cost_usd: float
    metrics: dict[str, float] = Field(default_factory=dict)
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: TrainingJobView) -> TrainingJobResponse:
        return cls(
            job_id=view.job_id,
            status=view.status,
            is_terminal=view.is_terminal,
            succeeded=view.succeeded,
            dataset_uri=view.dataset_uri,
            artifact_uri=view.artifact_uri,
            failure_reason=view.failure_reason,
            instance=view.instance,
            worst_case_cost_usd=view.worst_case_cost_usd,
            metrics=dict(view.metrics),
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class ReleaseResponse(BaseModel):
    fleet_id: str
    version: str
    channel: str
    verdict: str
    can_publish: bool
    artifact_bytes: int
    fleet_transfer_mib: float
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: ReleaseView) -> ReleaseResponse:
        return cls(
            fleet_id=view.fleet_id,
            version=view.version,
            channel=view.channel,
            verdict=view.verdict,
            can_publish=view.can_publish,
            artifact_bytes=view.artifact_bytes,
            fleet_transfer_mib=view.fleet_transfer_mib,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class RolloutResponse(BaseModel):
    rollout_id: str
    version: str
    previous_version: str
    status: str
    device_count: int
    succeeded: int
    failed: int
    unreachable: int
    coverage: float
    halt_reason: str
    current_wave: str
    history: list[dict[str, str]] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: RolloutView) -> RolloutResponse:
        return cls(
            rollout_id=view.rollout_id,
            version=view.version,
            previous_version=view.previous_version,
            status=view.status,
            device_count=view.device_count,
            succeeded=view.succeeded,
            failed=view.failed,
            unreachable=view.unreachable,
            coverage=view.coverage,
            halt_reason=view.halt_reason,
            current_wave=view.current_wave,
            history=[{"at": at, "what": what} for at, what in view.history],
            report=view.render(),
        )


class WaveResponse(BaseModel):
    rollout_id: str
    wave: str
    size: int
    succeeded: int
    failed: int
    unreachable: int
    pending: int
    failure_ratio: float
    halted: bool
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: WaveView) -> WaveResponse:
        return cls(
            rollout_id=view.rollout_id,
            wave=view.wave,
            size=view.size,
            succeeded=view.succeeded,
            failed=view.failed,
            unreachable=view.unreachable,
            pending=view.pending,
            failure_ratio=view.failure_ratio,
            halted=view.halted,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class LineageResponse(BaseModel):
    fleet_id: str
    device_id: str
    closed: bool
    verdict: str
    broken_stages: list[str] = Field(default_factory=list)
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: LineageView) -> LineageResponse:
        return cls(
            fleet_id=view.fleet_id,
            device_id=view.device_id,
            closed=view.closed,
            verdict=view.verdict,
            broken_stages=list(view.broken_stages),
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class ChannelResponse(BaseModel):
    fleet_id: str
    channels: str
