"""Model API 의 요청/응답 DTO 와 Domain 매퍼."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from application.model.dto import (
    ArchitectureProfileView,
    EvaluationView,
    ModelCertificateView,
    PreparationView,
    TensorSummaryView,
    TrainingCurveView,
    TrainingRunView,
)
from domain.model.acceptance import LatencyBudget, ModelAcceptancePolicy
from domain.model.architecture import ArchitectureKind, ModelArchitecture
from domain.model.curve import LearningPolicy, OverfittingPolicy
from domain.model.evaluation import EvaluationPolicy
from domain.model.protocol import EvaluationProtocol
from domain.model.tensor_spec import TensorLayout, TensorSpec
from domain.model.training_config import (
    EarlyStoppingRule,
    Optimizer,
    TrainingConfig,
)
from domain.model.windowing import WindowingPlan, WindowLabelPolicy
from interfaces.http.schemas.dataset import FindingResponse


# ---------------------------------------------------------------------------
# 요청
# ---------------------------------------------------------------------------
class ArchitectureRequest(BaseModel):
    kind: Literal["MLP", "CNN1D", "CNN2D"] = "CNN1D"
    input_shape: list[int] = Field(examples=[[30, 6]])
    class_count: int = Field(ge=2, examples=[3])
    hidden_channels: list[int] = Field(default_factory=lambda: [16, 32])
    kernel_size: int = 5
    dropout: float = 0.1
    layout: Literal["TIME_FIRST", "CHANNEL_FIRST", "CHANNEL_LAST"] = "TIME_FIRST"

    def to_domain(self) -> ModelArchitecture:
        return ModelArchitecture(
            kind=ArchitectureKind(self.kind),
            input_spec=TensorSpec(
                shape=tuple(self.input_shape), layout=TensorLayout(self.layout)
            ),
            class_count=self.class_count,
            hidden_channels=tuple(self.hidden_channels),
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )


class EarlyStoppingRequest(BaseModel):
    patience: int = 5
    min_delta: float = 1e-4
    monitor: Literal["validation_loss", "validation_accuracy"] = "validation_loss"

    def to_domain(self) -> EarlyStoppingRule:
        return EarlyStoppingRule(**self.model_dump())


class TrainingConfigRequest(BaseModel):
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: Literal["SGD", "ADAM"] = "ADAM"
    weight_decay: float = 0.0
    seed: int = 42
    class_weighted_loss: bool = False
    early_stopping: EarlyStoppingRequest | None = None

    def to_domain(self) -> TrainingConfig:
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            optimizer=Optimizer(self.optimizer),
            weight_decay=self.weight_decay,
            seed=self.seed,
            class_weighted_loss=self.class_weighted_loss,
            early_stopping=(
                self.early_stopping.to_domain() if self.early_stopping else None
            ),
        )


class WindowingRequest(BaseModel):
    window_length: int = Field(ge=1, examples=[30])
    stride: int = Field(ge=1, examples=[30])
    label_priority: list[tuple[str, float]] = Field(
        default_factory=lambda: [("FAULT", 0.3), ("OVERLOAD", 0.5)]
    )
    default_label: str = "NORMAL"

    def to_domain(self) -> WindowingPlan:
        return WindowingPlan(
            window_length=self.window_length,
            stride=self.stride,
            label_policy=WindowLabelPolicy(
                priority=tuple((name, ratio) for name, ratio in self.label_priority),
                default_label=self.default_label,
            ),
        )


class PrepareTrainingRunRequest(BaseModel):
    run_id: str = Field(examples=["run-2026-05-11"])
    dataset_id: str
    assessment_id: str | None = None
    architecture: ArchitectureRequest
    config: TrainingConfigRequest = Field(default_factory=TrainingConfigRequest)
    windowing: WindowingRequest
    require_gates: bool = True
    """False 로 두면 게이트를 통과하지 않은 데이터로도 준비된다. 실습용 탈출구다."""


class EvaluationPolicyRequest(BaseModel):
    min_accuracy_over_baseline: float = 0.03
    min_recall_per_class: float = 0.5
    critical_labels: list[str] = Field(default_factory=list)
    min_critical_recall: float = 0.7
    max_precision_recall_gap: float = 0.5

    def to_domain(self) -> EvaluationPolicy:
        return EvaluationPolicy(
            min_accuracy_over_baseline=self.min_accuracy_over_baseline,
            min_recall_per_class=self.min_recall_per_class,
            critical_labels=frozenset(self.critical_labels),
            min_critical_recall=self.min_critical_recall,
            max_precision_recall_gap=self.max_precision_recall_gap,
        )


class EvaluateRequest(BaseModel):
    split: Literal["train", "validation", "test"] = "test"
    policy: EvaluationPolicyRequest = Field(default_factory=EvaluationPolicyRequest)


class FieldEvaluateRequest(BaseModel):
    field_uri: str = Field(examples=["data/samples/plant_power_model_field.csv"])
    split_name: str = "field"
    policy: EvaluationPolicyRequest = Field(default_factory=EvaluationPolicyRequest)


class AcceptanceRequest(BaseModel):
    split: str = "test"
    evaluation: EvaluationPolicyRequest = Field(
        default_factory=EvaluationPolicyRequest
    )
    latency_p95_ms: float | None = None
    """현장의 사이클 타임이 정하는 숫자다. 없으면 지연시간을 보지 않는다."""

    min_train_loss_drop: float = 0.2
    max_generalization_gap: float = 0.10
    max_validation_evaluations: int = 20
    require_gates: bool = True

    def to_domain(self) -> ModelAcceptancePolicy:
        return ModelAcceptancePolicy(
            learning=LearningPolicy(min_train_loss_drop=self.min_train_loss_drop),
            overfitting=OverfittingPolicy(
                max_generalization_gap=self.max_generalization_gap
            ),
            evaluation=self.evaluation.to_domain(),
            protocol=EvaluationProtocol(
                max_validation_evaluations=self.max_validation_evaluations
            ),
            latency=(
                LatencyBudget(p95_ms=self.latency_p95_ms)
                if self.latency_p95_ms
                else None
            ),
            require_gates=self.require_gates,
        )


class ReopenTrainingRunRequest(BaseModel):
    reason: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# 응답
# ---------------------------------------------------------------------------
class TensorSummaryResponse(BaseModel):
    split: str
    sample_count: int
    sample_shape: list[int]
    class_counts: dict[str, int]
    feature_mean: float | None = None
    feature_std: float | None = None
    nan_count: int

    @classmethod
    def from_view(cls, view: TensorSummaryView) -> TensorSummaryResponse:
        return cls(
            split=view.split,
            sample_count=view.sample_count,
            sample_shape=list(view.sample_shape),
            class_counts=dict(view.class_counts),
            feature_mean=view.feature_mean,
            feature_std=view.feature_std,
            nan_count=view.nan_count,
        )


class PreparationResponse(BaseModel):
    run_id: str
    dataset_ref: str
    architecture: str
    windowing: str
    input_shape: list[int]
    batch_shape: list[int]
    bytes_per_batch: int
    summaries: list[TensorSummaryResponse]
    windowing_report: str
    findings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: PreparationView) -> PreparationResponse:
        return cls(
            run_id=view.run_id,
            dataset_ref=view.dataset_ref,
            architecture=view.architecture,
            windowing=view.windowing,
            input_shape=list(view.input_shape),
            batch_shape=list(view.batch_shape),
            bytes_per_batch=view.bytes_per_batch,
            summaries=[TensorSummaryResponse.from_view(s) for s in view.summaries],
            windowing_report=view.windowing_report,
            findings=[FindingResponse.from_view(f) for f in view.findings],
        )


class ArchitectureProfileResponse(BaseModel):
    parameter_count: int
    mac_count: int
    parameter_bytes: int
    heaviest_layer: str | None = None
    busiest_layer: str | None = None
    table: str

    @classmethod
    def from_view(cls, view: ArchitectureProfileView) -> ArchitectureProfileResponse:
        return cls(
            parameter_count=view.parameter_count,
            mac_count=view.mac_count,
            parameter_bytes=view.parameter_bytes,
            heaviest_layer=view.heaviest_layer,
            busiest_layer=view.busiest_layer,
            table=view.table,
        )


class TrainingRunResponse(BaseModel):
    run_id: str
    dataset_ref: str
    status: str
    architecture: str
    config: str
    windowing: str
    epoch_count: int
    best_epoch: int | None = None
    model_version_id: str | None = None
    verdict: str | None = None
    failure_reason: str | None = None
    evaluated_splits: list[str]

    @classmethod
    def from_view(cls, view: TrainingRunView) -> TrainingRunResponse:
        return cls(
            run_id=view.run_id,
            dataset_ref=view.dataset_ref,
            status=view.status,
            architecture=view.architecture,
            config=view.config,
            windowing=view.windowing,
            epoch_count=view.epoch_count,
            best_epoch=view.best_epoch,
            model_version_id=view.model_version_id,
            verdict=view.verdict,
            failure_reason=view.failure_reason,
            evaluated_splits=list(view.evaluated_splits),
        )


class EpochResponse(BaseModel):
    epoch: int
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float


class TrainingCurveResponse(BaseModel):
    run_id: str
    status: str
    epoch_count: int
    best_epoch: int | None = None
    train_loss_drop: float
    overfitting_epoch: int | None = None
    final_gap: float
    wasted_epochs: int
    total_seconds: float
    table: str
    findings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: TrainingCurveView) -> TrainingCurveResponse:
        return cls(
            run_id=view.run_id,
            status=view.status,
            epoch_count=view.epoch_count,
            best_epoch=view.best_epoch,
            train_loss_drop=view.train_loss_drop,
            overfitting_epoch=view.overfitting_epoch,
            final_gap=view.final_gap,
            wasted_epochs=view.wasted_epochs,
            total_seconds=view.total_seconds,
            table=view.table,
            findings=[FindingResponse.from_view(f) for f in view.findings],
        )


class EvaluationResponse(BaseModel):
    run_id: str
    split: str
    accuracy: float
    baseline_accuracy: float
    macro_recall: float
    macro_f1: float
    loss: float
    latency_ms_p50: float
    latency_ms_p95: float
    per_class: list[dict[str, float | str | int]]
    never_predicted: list[str]
    matrix: str
    findings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: EvaluationView) -> EvaluationResponse:
        return cls(
            run_id=view.run_id,
            split=view.split,
            accuracy=view.accuracy,
            baseline_accuracy=view.baseline_accuracy,
            macro_recall=view.macro_recall,
            macro_f1=view.macro_f1,
            loss=view.loss,
            latency_ms_p50=view.latency_ms_p50,
            latency_ms_p95=view.latency_ms_p95,
            per_class=[
                {
                    "label": label,
                    "support": support,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                }
                for label, support, recall, precision, f1 in view.per_class
            ],
            never_predicted=list(view.never_predicted),
            matrix=view.matrix,
            findings=[FindingResponse.from_view(f) for f in view.findings],
        )


class ModelCertificateResponse(BaseModel):
    run_id: str
    model_version_id: str
    verdict: str
    is_deployable: bool
    accuracy: float
    macro_recall: float
    latency_ms_p95: float
    blocking: list[FindingResponse]
    warnings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: ModelCertificateView) -> ModelCertificateResponse:
        return cls(
            run_id=view.run_id,
            model_version_id=view.model_version_id,
            verdict=view.verdict,
            is_deployable=view.is_deployable,
            accuracy=view.accuracy,
            macro_recall=view.macro_recall,
            latency_ms_p95=view.latency_ms_p95,
            blocking=[FindingResponse.from_view(f) for f in view.blocking],
            warnings=[FindingResponse.from_view(f) for f in view.warnings],
        )
