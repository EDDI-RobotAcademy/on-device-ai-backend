"""Optimization API 의 요청/응답 DTO 와 Domain 매퍼. (모듈 4)

HTTP DTO 는 Domain Model 이 아니다 (CLAUDE.md §10).
특히 여기서는 그 구분이 눈에 보인다 —
`MeasurementProtocol` 은 요청으로도 오고 응답에도 붙어 다닌다.
숫자만 돌려주면 "어떻게 쟀는지"가 사라지기 때문이다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from application.optimization.dto import (
    BenchmarkView,
    CandidateView,
    OptimizationRunView,
    RooflineView,
    SelectionView,
    TradeoffView,
)
from domain.optimization.benchmark import BenchmarkPolicy, MeasurementProtocol
from domain.optimization.conversion import EquivalencePolicy
from domain.optimization.roofline import DeviceCapability, RooflinePolicy
from domain.optimization.runtime import Precision, RuntimeTarget
from domain.optimization.selection import DeviceBudget, SelectionObjective
from interfaces.http.schemas.dataset import FindingResponse

RuntimeName = Literal["PYTORCH", "TORCHSCRIPT", "ONNX", "TFLITE"]
PrecisionName = Literal["FP32", "FP16", "INT8"]


# ---------------------------------------------------------------------------
# 요청
# ---------------------------------------------------------------------------
class MeasurementProtocolRequest(BaseModel):
    """어떻게 잴 것인가. 숫자와 항상 함께 다닌다. (실습 4-1)"""

    warmup_runs: int = Field(default=30, ge=0)
    measured_runs: int = Field(default=300, ge=1)
    batch_size: int = Field(default=1, ge=1)
    threads: int = Field(default=1, ge=1)

    def to_domain(self) -> MeasurementProtocol:
        return MeasurementProtocol(**self.model_dump())


class BenchmarkPolicyRequest(BaseModel):
    min_warmup_runs: int = 10
    min_measured_runs: int = 100
    max_jitter_ratio: float = 2.0
    require_single_thread: bool = True
    require_batch_one: bool = True

    def to_domain(self) -> BenchmarkPolicy:
        return BenchmarkPolicy(**self.model_dump())


class EquivalencePolicyRequest(BaseModel):
    max_abs_diff_fp32: float = 1e-4
    max_abs_diff_fp16: float = 1e-2
    min_agreement_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    min_sample_count: int = Field(default=32, ge=1)

    def to_domain(self) -> EquivalencePolicy:
        return EquivalencePolicy(**self.model_dump())


class StartOptimizationRunRequest(BaseModel):
    run_id: str = Field(examples=["opt-power"])
    training_run_id: str = Field(examples=["run-opt"])
    split: str = "test"
    require_accepted: bool = True


class BenchmarkBaselineRequest(BaseModel):
    protocol: MeasurementProtocolRequest = Field(
        default_factory=MeasurementProtocolRequest
    )
    policy: BenchmarkPolicyRequest = Field(default_factory=BenchmarkPolicyRequest)
    split: str = "test"


class ConvertModelRequest(BaseModel):
    runtime: RuntimeName = Field(examples=["ONNX"])
    precision: PrecisionName = "FP32"
    equivalence_samples: int = Field(default=128, ge=1)
    split: str = "test"
    protocol: MeasurementProtocolRequest = Field(
        default_factory=MeasurementProtocolRequest
    )
    equivalence_policy: EquivalencePolicyRequest = Field(
        default_factory=EquivalencePolicyRequest
    )
    benchmark_policy: BenchmarkPolicyRequest = Field(
        default_factory=BenchmarkPolicyRequest
    )

    def runtime_target(self) -> RuntimeTarget:
        return RuntimeTarget(self.runtime)

    def precision_target(self) -> Precision:
        return Precision(self.precision)


class DeviceCapabilityRequest(BaseModel):
    """데이터시트에서 오는 숫자다. 모델이 정하지 않는다."""

    name: str = Field(examples=["edge-mcu"])
    peak_gmac_per_second: float = Field(gt=0, examples=[2.0])
    memory_bandwidth_gb_per_second: float = Field(gt=0, examples=[1.6])

    def to_domain(self) -> DeviceCapability:
        return DeviceCapability(**self.model_dump())


class RooflinePolicyRequest(BaseModel):
    max_memory_bound_share: float = 0.5
    min_intensity_warning: float = 1.0

    def to_domain(self) -> RooflinePolicy:
        return RooflinePolicy(**self.model_dump())


class ProfileRooflineRequest(BaseModel):
    device: DeviceCapabilityRequest
    policy: RooflinePolicyRequest = Field(default_factory=RooflinePolicyRequest)


class DeviceBudgetRequest(BaseModel):
    """설비와 하드웨어가 정하는 것. (실습 4-10)"""

    name: str = Field(examples=["전력 감시 설비"])
    latency_p95_ms: float = Field(gt=0, examples=[1.0])
    storage_kib: float | None = Field(default=None, gt=0)
    activation_kib: float | None = Field(default=None, gt=0)
    min_macro_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    max_accuracy_drop: float = Field(default=0.02, ge=0.0, le=1.0)
    max_class_recall_drop: float = Field(default=0.05, ge=0.0, le=1.0)

    def to_domain(self) -> DeviceBudget:
        return DeviceBudget(**self.model_dump())


class SelectModelRequest(BaseModel):
    budget: DeviceBudgetRequest
    objective: Literal["ACCURACY", "LATENCY", "SIZE"] = "ACCURACY"
    equivalence: EquivalencePolicyRequest = Field(
        default_factory=EquivalencePolicyRequest
    )
    require_deployable_runtime: bool = True

    def objective_target(self) -> SelectionObjective:
        return SelectionObjective(self.objective)


class ReopenOptimizationRunRequest(BaseModel):
    reason: str = Field(min_length=1, examples=["예산이 20ms 로 바뀌었다"])


# ---------------------------------------------------------------------------
# 응답
# ---------------------------------------------------------------------------
class OptimizationRunResponse(BaseModel):
    run_id: str
    model_version_id: str
    status: str
    baseline_label: str | None = None
    candidate_labels: list[str] = Field(default_factory=list)
    rejections: list[dict[str, str]] = Field(default_factory=list)
    selected_label: str | None = None
    verdict: str | None = None

    @classmethod
    def from_view(cls, view: OptimizationRunView) -> OptimizationRunResponse:
        return cls(
            run_id=view.run_id,
            model_version_id=view.model_version_id,
            status=view.status,
            baseline_label=view.baseline_label,
            candidate_labels=list(view.candidate_labels),
            rejections=[
                {"label": label, "reason": reason} for label, reason in view.rejections
            ],
            selected_label=view.selected_label,
            verdict=view.verdict,
        )


class BenchmarkResponse(BaseModel):
    run_id: str
    label: str
    protocol: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    jitter_ratio: float
    size_bytes: int
    activation_bytes: int
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: BenchmarkView) -> BenchmarkResponse:
        return cls(
            run_id=view.run_id,
            label=view.label,
            protocol=view.protocol,
            p50_ms=view.p50_ms,
            p95_ms=view.p95_ms,
            p99_ms=view.p99_ms,
            jitter_ratio=view.jitter_ratio,
            size_bytes=view.size_bytes,
            activation_bytes=view.activation_bytes,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class CandidateResponse(BaseModel):
    run_id: str
    label: str
    artifact_id: str
    runtime: str
    precision: str
    size_bytes: int
    theoretical_weight_bytes: int
    overhead_bytes: int
    p50_ms: float
    p95_ms: float
    accuracy: float
    macro_recall: float
    macro_f1: float
    per_class_recall: dict[str, float] = Field(default_factory=dict)
    equivalence: str
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: CandidateView) -> CandidateResponse:
        return cls(
            run_id=view.run_id,
            label=view.label,
            artifact_id=view.artifact_id,
            runtime=view.runtime,
            precision=view.precision,
            size_bytes=view.size_bytes,
            theoretical_weight_bytes=view.theoretical_weight_bytes,
            overhead_bytes=view.overhead_bytes,
            p50_ms=view.p50_ms,
            p95_ms=view.p95_ms,
            accuracy=view.accuracy,
            macro_recall=view.macro_recall,
            macro_f1=view.macro_f1,
            per_class_recall=dict(view.per_class_recall),
            equivalence=view.equivalence,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class RooflineResponse(BaseModel):
    run_id: str
    device: str
    total_macs: int
    total_bytes_moved: int
    overall_intensity: float
    machine_balance: float
    dominant_bottleneck: str
    busiest_layer: str | None = None
    heaviest_traffic_layer: str | None = None
    findings: list[FindingResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: RooflineView) -> RooflineResponse:
        return cls(
            run_id=view.run_id,
            device=view.device,
            total_macs=view.total_macs,
            total_bytes_moved=view.total_bytes_moved,
            overall_intensity=view.overall_intensity,
            machine_balance=view.machine_balance,
            dominant_bottleneck=view.dominant_bottleneck,
            busiest_layer=view.busiest_layer,
            heaviest_traffic_layer=view.heaviest_traffic_layer,
            findings=[FindingResponse.from_view(f) for f in view.findings],
            report=view.render(),
        )


class TradeoffResponse(BaseModel):
    run_id: str
    fastest: str
    smallest: str
    most_accurate: str
    slower_than_baseline: list[str] = Field(default_factory=list)
    pareto_front: list[str] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: TradeoffView) -> TradeoffResponse:
        return cls(
            run_id=view.run_id,
            fastest=view.fastest,
            smallest=view.smallest,
            most_accurate=view.most_accurate,
            slower_than_baseline=list(view.slower_than_baseline),
            pareto_front=list(view.pareto_front),
            report=view.render(),
        )


class RejectedCandidateResponse(BaseModel):
    label: str
    reasons: list[str] = Field(default_factory=list)


class SelectionResponse(BaseModel):
    run_id: str
    verdict: str
    has_selection: bool
    selected_label: str | None = None
    selected_p95_ms: float | None = None
    selected_accuracy: float | None = None
    budget: str
    objective: str
    rejected: list[RejectedCandidateResponse] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view: SelectionView) -> SelectionResponse:
        return cls(
            run_id=view.run_id,
            verdict=view.verdict,
            has_selection=view.has_selection,
            selected_label=view.selected_label,
            selected_p95_ms=view.selected_p95_ms,
            selected_accuracy=view.selected_accuracy,
            budget=view.budget,
            objective=view.objective,
            rejected=[
                RejectedCandidateResponse(label=label, reasons=list(reasons))
                for label, reasons in view.rejected
            ],
            report=view.render(),
        )
