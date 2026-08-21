"""Dataset API 의 요청/응답 DTO 와 Domain 매퍼."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from application.data.dto import (
    DatasetProfileView,
    DatasetView,
    FindingView,
    InspectionView,
    ReadinessView,
)
from application.data.design_training_data import TrainingDesignView
from application.data.analyze_representativeness import RepresentativenessView
from application.data.partition_dataset import PartitionView
from domain.data.labeling import LabelDefinition, LabelPolicy, LabelSpace
from domain.data.partition import (
    PartitionPlan,
    PartitionPolicy,
    SplitRatio,
    SplitStrategy,
)
from domain.data.profile import FieldType
from domain.data.readiness import DEFAULT_REQUIRED_KINDS, ReadinessPolicy
from domain.data.representativeness import RepresentativenessPolicy
from domain.data.schema import DataSchema, FieldRole, FieldSpec, ValueRange
from domain.data.signal import SignalPlausibilityPolicy
from domain.data.source import DataSourceDescriptor, Modality, SourceFormat
from domain.data.time_axis import SamplingInterval, TimeAxisPolicy
from domain.data.training_spec import (
    ImageInputSpec,
    NormalizationMethod,
    NormalizationSpec,
    TrainingDataSpec,
    WindowSpec,
)


# ---------------------------------------------------------------------------
# 요청
# ---------------------------------------------------------------------------
class RegisterDatasetRequest(BaseModel):
    dataset_id: str = Field(examples=["plant-power-2026-03"])
    name: str = Field(examples=["3라인 주회로 전력 24시간"])
    uri: str = Field(examples=["data/samples/plant_power_raw.csv"])
    source_format: Literal["CSV", "PARQUET", "IMAGE_DIRECTORY"] = "CSV"
    modality: Literal["TIME_SERIES", "IMAGE", "TABULAR"] = "TIME_SERIES"
    collected_from: str = Field(examples=["LINE-3 / PM-MAIN-01"])

    def to_format(self) -> SourceFormat:
        return SourceFormat(self.source_format)

    def to_modality(self) -> Modality:
        return Modality(self.modality)


class ValueRangeRequest(BaseModel):
    minimum: float
    maximum: float

    def to_domain(self) -> ValueRange:
        return ValueRange(minimum=self.minimum, maximum=self.maximum)


class FieldSpecRequest(BaseModel):
    name: str
    type: Literal[
        "TIMESTAMP", "REAL", "INTEGER", "CATEGORY", "BOOLEAN", "TEXT", "IMAGE_REF"
    ]
    role: Literal[
        "TIME_INDEX", "FEATURE", "LABEL", "IDENTIFIER", "GROUP", "METADATA"
    ]
    unit: str | None = None
    required: bool = True
    value_range: ValueRangeRequest | None = None

    def to_domain(self) -> FieldSpec:
        return FieldSpec(
            name=self.name,
            type=FieldType(self.type),
            role=FieldRole(self.role),
            unit=self.unit,
            required=self.required,
            value_range=self.value_range.to_domain() if self.value_range else None,
        )


class DeclareSchemaRequest(BaseModel):
    fields: list[FieldSpecRequest]

    def to_domain(self) -> DataSchema:
        return DataSchema(fields=tuple(f.to_domain() for f in self.fields))


class SignalPolicyRequest(BaseModel):
    max_out_of_range_ratio: float = 0.0
    max_constant_run_ratio: float = 0.05
    min_constant_run_length: int = 10
    max_saturated_ratio: float = 0.02
    min_focus_score: float = 50.0
    max_unreadable_ratio: float = 0.0
    max_defocused_ratio: float = 0.05
    max_visual_duplicate_ratio: float = 0.02
    max_brightness_stddev: float = 40.0

    def to_domain(self) -> SignalPlausibilityPolicy:
        return SignalPlausibilityPolicy(**self.model_dump())


class TimeAxisPolicyRequest(BaseModel):
    expected_interval_seconds: float = Field(gt=0, examples=[10.0])
    interval_tolerance_ratio: float = 0.2
    max_gap_multiplier: float = 3.0
    max_duplicate_ratio: float = 0.0
    allow_out_of_order: bool = False
    min_coverage_ratio: float = 0.95

    def to_domain(self) -> TimeAxisPolicy:
        return TimeAxisPolicy(
            expected_interval=SamplingInterval(seconds=self.expected_interval_seconds),
            interval_tolerance_ratio=self.interval_tolerance_ratio,
            max_gap_multiplier=self.max_gap_multiplier,
            max_duplicate_ratio=self.max_duplicate_ratio,
            allow_out_of_order=self.allow_out_of_order,
            min_coverage_ratio=self.min_coverage_ratio,
        )


class LabelDefinitionRequest(BaseModel):
    name: str
    meaning: str
    decided_by: str
    examples: list[str] = Field(default_factory=list)

    def to_domain(self) -> LabelDefinition:
        return LabelDefinition(
            name=self.name,
            meaning=self.meaning,
            decided_by=self.decided_by,
            examples=tuple(self.examples),
        )


class LabelPolicyRequest(BaseModel):
    min_agreement_ratio: float = 0.9
    min_samples_per_class: int = 30
    max_unlabeled_ratio: float = 0.0
    max_imbalance_ratio: float = 10.0
    require_cross_review: bool = True

    def to_domain(self) -> LabelPolicy:
        return LabelPolicy(**self.model_dump())


class DefineLabelSpaceRequest(BaseModel):
    field_name: str
    definitions: list[LabelDefinitionRequest]
    policy: LabelPolicyRequest = Field(default_factory=LabelPolicyRequest)

    def to_domain(self) -> LabelSpace:
        return LabelSpace(
            field_name=self.field_name,
            definitions=tuple(d.to_domain() for d in self.definitions),
        )


class WindowSpecRequest(BaseModel):
    length: int = Field(ge=1)
    stride: int = Field(ge=1)
    interval_seconds: float = Field(gt=0)

    def to_domain(self) -> WindowSpec:
        return WindowSpec(
            length=self.length,
            stride=self.stride,
            interval=SamplingInterval(seconds=self.interval_seconds),
        )


class ImageInputSpecRequest(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    channels: Literal[1, 3] = 3

    def to_domain(self) -> ImageInputSpec:
        return ImageInputSpec(
            width=self.width, height=self.height, channels=self.channels
        )


class NormalizationRequest(BaseModel):
    method: Literal["NONE", "ZSCORE", "MINMAX"] = "NONE"
    fitted_on: str = "train"
    statistics: dict[str, tuple[float, float]] = Field(default_factory=dict)

    def to_domain(self) -> NormalizationSpec:
        return NormalizationSpec(
            method=NormalizationMethod(self.method),
            fitted_on=self.fitted_on,
            statistics=dict(self.statistics),
        )


class DesignTrainingDataRequest(BaseModel):
    feature_fields: list[str]
    label_field: str
    window: WindowSpecRequest | None = None
    image: ImageInputSpecRequest | None = None
    normalization: NormalizationRequest = Field(default_factory=NormalizationRequest)
    fit_normalization: bool = False
    """True 면 서버가 train 분할에서 정규화 통계를 직접 계산한다. 분할이 먼저 있어야 한다."""

    def to_domain(self, schema: DataSchema) -> TrainingDataSpec:
        return TrainingDataSpec(
            schema=schema,
            feature_fields=tuple(self.feature_fields),
            label_field=self.label_field,
            window=self.window.to_domain() if self.window else None,
            image=self.image.to_domain() if self.image else None,
            normalization=self.normalization.to_domain(),
        )


class PartitionRequest(BaseModel):
    strategy: Literal["RANDOM", "TIME_ORDERED", "GROUP_HOLDOUT"]
    train: float = 0.7
    validation: float = 0.15
    test: float = 0.15
    time_field: str | None = None
    group_field: str | None = None
    seed: int = 42
    min_samples_per_split: int = 1
    ratio_tolerance: float = 0.05
    max_class_ratio_gap: float = 0.1

    def to_plan(self) -> PartitionPlan:
        return PartitionPlan(
            strategy=SplitStrategy(self.strategy),
            ratio=SplitRatio.of(self.train, self.validation, self.test),
            time_field=self.time_field,
            group_field=self.group_field,
            seed=self.seed,
        )

    def to_policy(self) -> PartitionPolicy:
        return PartitionPolicy(
            min_samples_per_split=self.min_samples_per_split,
            ratio_tolerance=self.ratio_tolerance,
            max_class_ratio_gap=self.max_class_ratio_gap,
        )


class RepresentativenessRequest(BaseModel):
    observed_uri: str
    observed_collected_from: str
    observed_format: Literal["CSV", "PARQUET"] = "CSV"
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.25
    min_coverage_ratio: float = 0.9
    max_unseen_category_ratio: float = 0.05
    min_observed_sample_count: int = 100

    def to_source(self, modality: Modality) -> DataSourceDescriptor:
        return DataSourceDescriptor(
            uri=self.observed_uri,
            format=SourceFormat(self.observed_format),
            modality=modality,
            collected_from=self.observed_collected_from,
        )

    def to_policy(self) -> RepresentativenessPolicy:
        return RepresentativenessPolicy(
            psi_warning_threshold=self.psi_warning_threshold,
            psi_critical_threshold=self.psi_critical_threshold,
            min_coverage_ratio=self.min_coverage_ratio,
            max_unseen_category_ratio=self.max_unseen_category_ratio,
            min_observed_sample_count=self.min_observed_sample_count,
        )


class ReadinessRequest(BaseModel):
    required_kinds: list[str] | None = None
    allow_warnings: bool = True
    max_warning_count: int = 10

    def to_policy(self) -> ReadinessPolicy:
        from domain.data.inspection import InspectionKind

        kinds = (
            frozenset(InspectionKind(k) for k in self.required_kinds)
            if self.required_kinds
            else DEFAULT_REQUIRED_KINDS
        )
        return ReadinessPolicy(
            required_kinds=kinds,
            allow_warnings=self.allow_warnings,
            max_warning_count=self.max_warning_count,
        )


class ReopenRequest(BaseModel):
    reason: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# 응답
# ---------------------------------------------------------------------------
class FindingResponse(BaseModel):
    code: str
    message: str
    severity: str
    subject: str | None = None
    measured: float | None = None
    threshold: float | None = None

    @classmethod
    def from_view(cls, view: FindingView) -> FindingResponse:
        return cls(**vars_of(view))


class InspectionResponse(BaseModel):
    dataset_id: str
    kind: str
    verdict: str
    findings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: InspectionView) -> InspectionResponse:
        return cls(
            dataset_id=view.dataset_id,
            kind=view.kind,
            verdict=view.verdict,
            findings=[FindingResponse.from_view(f) for f in view.findings],
        )


class ColumnProfileResponse(BaseModel):
    name: str
    inferred_type: str
    total_count: int
    missing_count: int
    missing_ratio: float
    distinct_count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    stddev: float | None = None
    sample_values: list[str] = Field(default_factory=list)


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    row_count: int
    byte_size: int
    columns: list[ColumnProfileResponse]

    @classmethod
    def from_view(cls, view: DatasetProfileView) -> DatasetProfileResponse:
        return cls(
            dataset_id=view.dataset_id,
            row_count=view.row_count,
            byte_size=view.byte_size,
            columns=[
                ColumnProfileResponse(
                    name=c.name,
                    inferred_type=c.inferred_type,
                    total_count=c.total_count,
                    missing_count=c.missing_count,
                    missing_ratio=c.missing_ratio,
                    distinct_count=c.distinct_count,
                    minimum=c.minimum,
                    maximum=c.maximum,
                    mean=c.mean,
                    stddev=c.stddev,
                    sample_values=list(c.sample_values),
                )
                for c in view.columns
            ],
        )


class SchemaDraftResponse(BaseModel):
    dataset_id: str
    fields: list[dict[str, str]]
    undecided_fields: list[str]


class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    status: str
    modality: str
    collected_from: str
    uri: str
    row_count: int | None = None
    field_count: int | None = None
    label_class_count: int | None = None
    input_shape: list[int] | None = None
    verdict: str | None = None
    inspections: list[InspectionResponse] = Field(default_factory=list)

    @classmethod
    def from_view(cls, view: DatasetView) -> DatasetResponse:
        return cls(
            dataset_id=view.dataset_id,
            name=view.name,
            status=view.status,
            modality=view.modality,
            collected_from=view.collected_from,
            uri=view.uri,
            row_count=view.row_count,
            field_count=view.field_count,
            label_class_count=view.label_class_count,
            input_shape=list(view.input_shape) if view.input_shape else None,
            verdict=view.verdict,
            inspections=[InspectionResponse.from_view(i) for i in view.inspections],
        )


class TrainingDesignResponse(BaseModel):
    dataset_id: str
    input_shape: list[int]
    input_element_count: int
    feature_fields: list[str]
    label_field: str
    window_seconds: float | None = None
    normalization_method: str
    normalization_fitted_on: str
    inspection: InspectionResponse

    @classmethod
    def from_view(cls, view: TrainingDesignView) -> TrainingDesignResponse:
        return cls(
            dataset_id=view.dataset_id,
            input_shape=list(view.input_shape),
            input_element_count=view.input_element_count,
            feature_fields=list(view.feature_fields),
            label_field=view.label_field,
            window_seconds=view.window_seconds,
            normalization_method=view.normalization_method,
            normalization_fitted_on=view.normalization_fitted_on,
            inspection=InspectionResponse.from_view(view.inspection),
        )


class PartitionResponse(BaseModel):
    dataset_id: str
    strategy: str
    train_count: int
    validation_count: int
    test_count: int
    overlapping_group_count: int
    time_overlap_seconds: float
    inspection: InspectionResponse

    @classmethod
    def from_view(cls, view: PartitionView) -> PartitionResponse:
        return cls(
            dataset_id=view.dataset_id,
            strategy=view.strategy,
            train_count=view.train_count,
            validation_count=view.validation_count,
            test_count=view.test_count,
            overlapping_group_count=view.overlapping_group_count,
            time_overlap_seconds=view.time_overlap_seconds,
            inspection=InspectionResponse.from_view(view.inspection),
        )


class RepresentativenessResponse(BaseModel):
    dataset_id: str
    observed_uri: str
    worst_field: str | None = None
    worst_psi: float
    field_psi: list[dict[str, float | str]]
    inspection: InspectionResponse

    @classmethod
    def from_view(cls, view: RepresentativenessView) -> RepresentativenessResponse:
        return cls(
            dataset_id=view.dataset_id,
            observed_uri=view.observed_uri,
            worst_field=view.worst_field,
            worst_psi=view.worst_psi,
            field_psi=[
                {"field": name, "psi": psi, "coverage_ratio": coverage}
                for name, psi, coverage in view.field_psi
            ],
            inspection=InspectionResponse.from_view(view.inspection),
        )


class ReadinessResponse(BaseModel):
    dataset_id: str
    verdict: str
    is_ready: bool
    evaluated_kinds: list[str]
    missing_kinds: list[str]
    blocking: list[FindingResponse]
    warnings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: ReadinessView) -> ReadinessResponse:
        return cls(
            dataset_id=view.dataset_id,
            verdict=view.verdict,
            is_ready=view.is_ready,
            evaluated_kinds=list(view.evaluated_kinds),
            missing_kinds=list(view.missing_kinds),
            blocking=[FindingResponse.from_view(f) for f in view.blocking],
            warnings=[FindingResponse.from_view(f) for f in view.warnings],
        )


def vars_of(view: FindingView) -> dict[str, object]:
    return {
        "code": view.code,
        "message": view.message,
        "severity": view.severity,
        "subject": view.subject,
        "measured": view.measured,
        "threshold": view.threshold,
    }
