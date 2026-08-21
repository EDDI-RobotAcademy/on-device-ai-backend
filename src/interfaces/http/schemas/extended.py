"""모듈 1~6 확장 실습의 HTTP DTO. (실습 1-11, 2-11, 3-11~3-15, 4-11~4-14, 5-12~5-15, 6-12~6-14)

Pydantic Schema 와 Domain Model 은 다른 것이다 (CLAUDE.md §10).
여기 있는 것은 전부 **밖으로 나가는 모양**이고, `from_view` 로만 만들어진다.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass

from pydantic import BaseModel, Field

from application.data.dto import FindingView


class FindingOut(BaseModel):
    code: str
    message: str
    severity: str
    subject: str | None = None
    measured: float | None = None
    threshold: float | None = None

    @classmethod
    def of(cls, view: FindingView) -> FindingOut:
        return cls(
            code=view.code,
            message=view.message,
            severity=view.severity,
            subject=view.subject,
            measured=view.measured,
            threshold=view.threshold,
        )


def _findings(views) -> list[FindingOut]:  # noqa: ANN001
    return [FindingOut.of(f) for f in views]


def _plain(view) -> dict:  # noqa: ANN001
    """slots dataclass 인 View 를 dict 로 편다. `vars()` 는 slots 에 안 통한다."""
    if is_dataclass(view):
        return {f.name: getattr(view, f.name) for f in fields(view)}
    return dict(view)


# ---------------------------------------------------------------------------
# 1-11 수집 설계
# ---------------------------------------------------------------------------
class SamplingPlanIn(BaseModel):
    interval_seconds: float = Field(gt=0)
    value_resolution: float = Field(default=0.0, ge=0)
    retention_days: int = Field(default=30, ge=1)
    bytes_per_sample: int = Field(default=64, ge=1)


class DesignSamplingRequest(BaseModel):
    plans: list[SamplingPlanIn] = Field(min_length=2)
    normal_label: str = "NORMAL"
    value_field: str | None = None
    target_event_seconds: float = Field(default=60.0, gt=0)
    min_samples_per_event: float = Field(default=5.0, gt=0)


class SamplingPlanOut(BaseModel):
    interval_seconds: float
    value_resolution: float
    retention_days: int
    row_count: int
    event_run_count: int
    lost_event_runs: int
    event_ratio: float
    bytes_per_day: float
    verdict: str


class SamplingTradeoffResponse(BaseModel):
    dataset_id: str
    plans: list[SamplingPlanOut]
    cheapest_acceptable: str | None
    acceptable_count: int
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> SamplingTradeoffResponse:  # noqa: ANN001
        return cls(
            dataset_id=view.dataset_id,
            plans=[SamplingPlanOut(**_plain(p)) for p in view.plans],
            cheapest_acceptable=view.cheapest_acceptable,
            acceptable_count=view.acceptable_count,
            findings=_findings(view.findings),
            report=view.render(),
        )


# ---------------------------------------------------------------------------
# 2-11 불균형 완화
# ---------------------------------------------------------------------------
class RebalancingPlanIn(BaseModel):
    strategy: str
    target_ratio: float = Field(default=1.0, gt=0, le=1)
    applied_after_split: bool = True


class CompareRebalancingRequest(BaseModel):
    plans: list[RebalancingPlanIn] = Field(min_length=1)


class RebalancingOutcomeOut(BaseModel):
    strategy: str
    plan: str
    total_before: int
    total_after: int
    imbalance_before: float
    imbalance_after: float
    duplicated_rows: int
    discarded_rows: int
    synthesized_rows: int
    distinct_minority_samples: int
    information_gain: int
    verdict: str


class RebalancingComparisonResponse(BaseModel):
    dataset_id: str
    outcomes: list[RebalancingOutcomeOut]
    safe_strategies: list[str]
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> RebalancingComparisonResponse:  # noqa: ANN001
        return cls(
            dataset_id=view.dataset_id,
            outcomes=[RebalancingOutcomeOut(**_plain(o)) for o in view.outcomes],
            safe_strategies=list(view.safe_strategies),
            findings=_findings(view.findings),
            report=view.render(),
        )


# ---------------------------------------------------------------------------
# 3-11 이미지 학습
# ---------------------------------------------------------------------------
class ImageSpecIn(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(default=3)


class PrepareImageTrainingRunRequest(BaseModel):
    run_id: str
    dataset_ref: str
    root_uri: str
    spec: ImageSpecIn
    class_count: int = Field(ge=2)
    hidden_channels: list[int] = Field(default_factory=lambda: [16, 32])
    kernel_size: int = 3
    pooling: str = "AVERAGE"
    epochs: int = Field(default=25, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=3e-3, gt=0)
    seed: int = 42
    require_gates: bool = True


# ---------------------------------------------------------------------------
# 3-12 / 3-14 / 3-15 실험 비교
# ---------------------------------------------------------------------------
class TrialIn(BaseModel):
    run_id: str
    label: str
    knobs: dict[str, str] = Field(min_length=1)
    split: str = "test"


class CompareExperimentsRequest(BaseModel):
    name: str
    trials: list[TrialIn] = Field(min_length=1)
    metric: str = "macro_f1"
    noise_band: float = Field(default=0.02, ge=0)
    min_evaluated_samples: int = Field(default=60, ge=1)


class ExperimentTrialOut(BaseModel):
    label: str
    knobs: str
    accuracy: float
    macro_recall: float
    macro_f1: float
    loss: float
    latency_ms_p50: float
    parameter_count: int
    epochs: int
    evaluated_samples: int
    seed: int
    data_ref: str


class ExperimentBoardResponse(BaseModel):
    name: str
    metric: str
    trials: list[ExperimentTrialOut]
    best_label: str | None
    gap_to_runner_up: float
    spread: float
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> ExperimentBoardResponse:  # noqa: ANN001
        return cls(
            name=view.name,
            metric=view.metric,
            trials=[ExperimentTrialOut(**_plain(t)) for t in view.trials],
            best_label=view.best_label,
            gap_to_runner_up=view.gap_to_runner_up,
            spread=view.spread,
            findings=_findings(view.findings),
            report=view.render(),
        )


# ---------------------------------------------------------------------------
# 3-13 통계 기준선
# ---------------------------------------------------------------------------
class CompareWithBaselineRequest(BaseModel):
    method: str = "THREE_SIGMA"
    threshold: float = Field(default=3.0, gt=0)
    min_flagged_ratio: float = Field(default=0.2, gt=0, le=1)
    normal_label: str = "NORMAL"
    split: str = "test"
    min_recall_gain: float = Field(default=0.05, ge=0)


class BaselineComparisonResponse(BaseModel):
    run_id: str
    detector: str
    statistical_recall: float
    statistical_precision: float
    model_recall: float
    model_precision: float
    recall_gain: float
    precision_gain: float
    model_type_accuracy: float
    type_count: int
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> BaselineComparisonResponse:  # noqa: ANN001
        return cls(
            run_id=view.run_id,
            detector=view.detector,
            statistical_recall=view.statistical_recall,
            statistical_precision=view.statistical_precision,
            model_recall=view.model_recall,
            model_precision=view.model_precision,
            recall_gain=view.recall_gain,
            precision_gain=view.precision_gain,
            model_type_accuracy=view.model_type_accuracy,
            type_count=view.type_count,
            findings=_findings(view.findings),
            report=view.render(),
        )


# ---------------------------------------------------------------------------
# 4-11 구조 축소
# ---------------------------------------------------------------------------
class ReductionIn(BaseModel):
    label: str
    kind: str
    ratio: float = Field(gt=0, lt=1)
    fine_tuned: bool = False


class ReduceStructureRequest(BaseModel):
    reductions: list[ReductionIn] = Field(min_length=1)
    split: str = "test"
    fine_tune_epochs: int = Field(default=25, ge=1)
    max_accuracy_drop: float = Field(default=0.03, ge=0)


class ReductionOutcomeOut(BaseModel):
    label: str
    reduction: str
    parameter_count_before: int
    parameter_count_after: int
    nonzero_parameter_count: int
    sparsity: float
    mac_count_before: int
    mac_count_after: int
    mac_reduction: float
    size_bytes_before: int
    size_bytes_after: int
    size_reduction: float
    accuracy_before: float
    accuracy_after: float
    accuracy_drop: float
    verdict: str


class ReductionComparisonResponse(BaseModel):
    run_id: str
    outcomes: list[ReductionOutcomeOut]
    usable: list[str]
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> ReductionComparisonResponse:  # noqa: ANN001
        return cls(
            run_id=view.run_id,
            outcomes=[ReductionOutcomeOut(**_plain(o)) for o in view.outcomes],
            usable=list(view.usable),
            findings=_findings(view.findings),
            report=view.render(),
        )


# ---------------------------------------------------------------------------
# 4-12 PTQ vs QAT
# ---------------------------------------------------------------------------
class CompareQuantizationRequest(BaseModel):
    bits: int = Field(default=8, ge=2, le=16)
    split: str = "test"
    epochs: int = Field(default=12, ge=1)
    per_channel: bool = True


class QuantizationOutcomeOut(BaseModel):
    label: str
    approach: str
    bits: int
    baseline_accuracy: float
    quantized_accuracy: float
    accuracy_drop: float
    quantized_macro_recall: float
    training_seconds: float
    weight_bytes: int


class QuantizationComparisonResponse(BaseModel):
    run_id: str
    bits: int
    post_training: QuantizationOutcomeOut
    quantization_aware: QuantizationOutcomeOut
    recovered: float
    extra_training_seconds: float
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> QuantizationComparisonResponse:  # noqa: ANN001
        return cls(
            run_id=view.run_id,
            bits=view.bits,
            post_training=QuantizationOutcomeOut(**vars(view.post_training)),
            quantization_aware=QuantizationOutcomeOut(**vars(view.quantization_aware)),
            recovered=view.recovered,
            extra_training_seconds=view.extra_training_seconds,
            findings=_findings(view.findings),
            report=view.render(),
        )


# ---------------------------------------------------------------------------
# 4-13 / 4-14 자원과 배치
# ---------------------------------------------------------------------------
class MeasureResourcesRequest(BaseModel):
    labels: list[str] = Field(default_factory=list)
    warmup_runs: int = Field(default=20, ge=0)
    measured_runs: int = Field(default=150, ge=1)
    batch_size: int = Field(default=1, ge=1)
    threads: int = Field(default=1, ge=1)
    max_rss_bytes: int = Field(default=256 * 1024 * 1024, ge=1)
    max_cores: float = Field(default=1.0, gt=0)


class ResourceUsageResponse(BaseModel):
    label: str
    peak_rss_bytes: int
    model_rss_bytes: int
    cpu_utilization: float
    artifact_bytes: int
    rss_to_artifact_ratio: float
    verdict: str
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> ResourceUsageResponse:  # noqa: ANN001
        return cls(
            label=view.label,
            peak_rss_bytes=view.peak_rss_bytes,
            model_rss_bytes=view.model_rss_bytes,
            cpu_utilization=view.cpu_utilization,
            artifact_bytes=view.artifact_bytes,
            rss_to_artifact_ratio=view.rss_to_artifact_ratio,
            verdict=view.verdict,
            findings=_findings(view.findings),
            report=view.render(),
        )


class ScaleBatchRequest(BaseModel):
    label: str
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 4, 16, 64])
    warmup_runs: int = Field(default=20, ge=0)
    measured_runs: int = Field(default=120, ge=1)
    cycle_time_ms: float = Field(default=30.0, gt=0)


class BatchScalingResponse(BaseModel):
    label: str
    cycle_time_ms: float
    batch_sizes: list[int]
    per_sample_ms: list[float]
    first_answer_ms: list[float]
    throughput_gain: float
    latency_cost: float
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> BatchScalingResponse:  # noqa: ANN001
        return cls(
            label=view.label,
            cycle_time_ms=view.cycle_time_ms,
            batch_sizes=list(view.batch_sizes),
            per_sample_ms=list(view.per_sample_ms),
            first_answer_ms=list(view.first_answer_ms),
            throughput_gain=view.throughput_gain,
            latency_cost=view.latency_cost,
            findings=_findings(view.findings),
            report=view.render(),
        )


# ---------------------------------------------------------------------------
# 6-12 / 6-13 / 6-14
# ---------------------------------------------------------------------------
class ExperimentRecordIn(BaseModel):
    trial_id: str
    dataset_version: str = ""
    code_version: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_uri: str = ""
    created_at: str = ""


class RecordExperimentRequest(BaseModel):
    records: list[ExperimentRecordIn] = Field(min_length=1)


class ExperimentLedgerResponse(BaseModel):
    experiment_id: str
    metric: str
    trial_count: int
    reproducible_count: int
    best_trial_id: str | None
    missing_artifacts: list[str]
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> ExperimentLedgerResponse:  # noqa: ANN001
        return cls(
            experiment_id=view.experiment_id,
            metric=view.metric,
            trial_count=view.trial_count,
            reproducible_count=view.reproducible_count,
            best_trial_id=view.best_trial_id,
            missing_artifacts=list(view.missing_artifacts),
            findings=_findings(view.findings),
            report=view.render(),
        )


class AccessStatementIn(BaseModel):
    sid: str
    effect: str
    principal: str
    actions: list[str] = Field(min_length=1)
    resources: list[str] = Field(min_length=1)


class GovernStorageRequest(BaseModel):
    versioning: bool = True
    encryption: str | None = "AES256"
    block_public_access: bool = True
    expiration_days: int | None = 365
    statements: list[AccessStatementIn] = Field(default_factory=list)
    version_prefix: str = ""


class BucketGovernanceResponse(BaseModel):
    bucket: str
    versioning_enabled: bool
    encryption_algorithm: str | None
    public_access_blocked: bool
    lifecycle_expiration_days: int | None
    statement_count: int
    overwritten_keys: list[str]
    verdict: str
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> BucketGovernanceResponse:  # noqa: ANN001
        return cls(
            bucket=view.bucket,
            versioning_enabled=view.versioning_enabled,
            encryption_algorithm=view.encryption_algorithm,
            public_access_blocked=view.public_access_blocked,
            lifecycle_expiration_days=view.lifecycle_expiration_days,
            statement_count=view.statement_count,
            overwritten_keys=list(view.overwritten_keys),
            verdict=view.verdict,
            findings=_findings(view.findings),
            report=view.render(),
        )


class EndpointVariantIn(BaseModel):
    name: str
    model_reference: str
    instance_type: str = "ml.m5.large"
    instance_count: int = Field(default=1, ge=1)
    weight: float = Field(default=1.0, gt=0)
    hourly_cost_usd: float = Field(default=0.115, ge=0)


class DeployEndpointRequest(BaseModel):
    name: str
    variants: list[EndpointVariantIn] = Field(min_length=1)
    cycle_time_ms: float = Field(gt=0)
    network_round_trip_ms: float = Field(ge=0)
    inference_ms: float = Field(default=5.0, ge=0)
    offline_tolerance_minutes: float = Field(default=0.0, ge=0)
    requests_per_hour: int = Field(default=0, ge=0)
    image_uri: str = "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/serve:1"


class EndpointResponse(BaseModel):
    name: str
    status: str
    variants: list[tuple[str, float]]
    instance_count: int
    monthly_cost_usd: float
    total_latency_ms: float
    cycle_time_ms: float
    verdict: str
    findings: list[FindingOut] = Field(default_factory=list)
    report: str

    @classmethod
    def from_view(cls, view) -> EndpointResponse:  # noqa: ANN001
        return cls(
            name=view.name,
            status=view.status,
            variants=[(n, w) for n, w in view.variants],
            instance_count=view.instance_count,
            monthly_cost_usd=view.monthly_cost_usd,
            total_latency_ms=view.total_latency_ms,
            cycle_time_ms=view.cycle_time_ms,
            verdict=view.verdict,
            findings=_findings(view.findings),
            report=view.render(),
        )
