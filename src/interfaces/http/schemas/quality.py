"""Data Quality API 의 요청/응답 DTO 와 Domain 매퍼."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from application.data_quality.dto import (
    AssessmentView,
    DimensionView,
    QualityComparisonView,
    QualityGateView,
    QualityScoreView,
    TrainingImpactView,
)
from domain.data_quality.balance import BalancePolicy
from domain.data_quality.completeness import CompletenessPolicy
from domain.data_quality.dimensions import QualityDimension
from domain.data_quality.gate import DEFAULT_WEIGHTS, QualityGatePolicy
from domain.data_quality.label_quality import (
    LabelConsistencyRule,
    LabelQualityPolicy,
)
from domain.data_quality.noise import NoisePolicy
from domain.data_quality.remediation import RemediationAction, RemediationKind
from domain.data_quality.uniqueness import UniquenessPolicy
from domain.data_quality.validity import ValidityPolicy
from interfaces.http.schemas.dataset import FindingResponse


# ---------------------------------------------------------------------------
# 요청
# ---------------------------------------------------------------------------
class StartAssessmentRequest(BaseModel):
    assessment_id: str = Field(examples=["qa-2026-04-06"])


class CompletenessPolicyRequest(BaseModel):
    max_missing_ratio: float = 0.02
    max_missing_run_ratio: float = 0.01
    min_missing_run_length: int = 10
    max_concentration_ratio: float = 0.6
    max_repeated_value_ratio: float = 0.01
    max_scattered_run_length: float = 2.0

    def to_domain(self) -> CompletenessPolicy:
        return CompletenessPolicy(**self.model_dump())


class ValidityPolicyRequest(BaseModel):
    max_outlier_ratio: float = 0.01
    max_out_of_range_ratio: float = 0.0
    max_rate_violation_ratio: float = 0.005
    max_masking_gap_ratio: float = 0.005
    max_normal_label_outlier_share: float = 0.7
    normal_label: str | None = "NORMAL"
    """무엇을 '아무 일도 없었음'으로 볼 것인가. 라인마다 라벨 이름이 다르다."""

    def to_domain(self) -> ValidityPolicy:
        return ValidityPolicy(**self.model_dump())


class LabelRuleRequest(BaseModel):
    label: str
    field_name: str
    expected_min: float | None = None
    expected_max: float | None = None
    description: str

    def to_domain(self) -> LabelConsistencyRule:
        return LabelConsistencyRule(
            label=self.label,
            field_name=self.field_name,
            expected_min=self.expected_min,
            expected_max=self.expected_max,
            description=self.description,
        )


class LabelQualityPolicyRequest(BaseModel):
    max_violation_ratio: float = 0.005
    max_conflict_ratio: float = 0.0
    min_accuracy_ceiling: float = 0.98

    def to_domain(self) -> LabelQualityPolicy:
        return LabelQualityPolicy(**self.model_dump())


class MeasureLabelQualityRequest(BaseModel):
    rules: list[LabelRuleRequest] = Field(default_factory=list)
    policy: LabelQualityPolicyRequest = Field(
        default_factory=LabelQualityPolicyRequest
    )

    def to_rules(self) -> tuple[LabelConsistencyRule, ...]:
        return tuple(r.to_domain() for r in self.rules)


class BalancePolicyRequest(BaseModel):
    max_imbalance_ratio: float = 20.0
    min_minority_count: int = 100
    min_expected_minority_in_test: float = 20.0
    max_baseline_accuracy: float = 0.95
    min_class_count: int = 2

    def to_domain(self) -> BalancePolicy:
        return BalancePolicy(**self.model_dump())


class NoisePolicyRequest(BaseModel):
    min_snr_db: float = 20.0
    max_reversal_ratio: float = 0.75
    min_high_frequency_ratio: float = 1e-5

    def to_domain(self) -> NoisePolicy:
        return NoisePolicy(**self.model_dump())


class UniquenessPolicyRequest(BaseModel):
    max_exact_duplicate_ratio: float = 0.005
    max_near_duplicate_ratio: float = 0.05
    max_conflicting_label_ratio: float = 0.0
    max_inflation_ratio: float = 1.05

    def to_domain(self) -> UniquenessPolicy:
        return UniquenessPolicy(**self.model_dump())


class GatePolicyRequest(BaseModel):
    weights: dict[str, float] | None = None
    minimum_overall_score: float = 80.0
    minimum_dimension_score: float = 50.0
    required_dimensions: list[str] | None = None
    blocking_dimensions: list[str] | None = None

    def to_domain(self) -> QualityGatePolicy:
        weights = (
            {QualityDimension(k): v for k, v in self.weights.items()}
            if self.weights
            else dict(DEFAULT_WEIGHTS)
        )
        required = (
            frozenset(QualityDimension(d) for d in self.required_dimensions)
            if self.required_dimensions is not None
            else frozenset(QualityDimension)
        )
        blocking = (
            frozenset(QualityDimension(d) for d in self.blocking_dimensions)
            if self.blocking_dimensions is not None
            else frozenset(
                {QualityDimension.LABEL_QUALITY, QualityDimension.COMPLETENESS}
            )
        )
        return QualityGatePolicy(
            weights=weights,
            minimum_overall_score=self.minimum_overall_score,
            minimum_dimension_score=self.minimum_dimension_score,
            required_dimensions=required,
            blocking_dimensions=blocking,
        )


class ScoreQualityRequest(BaseModel):
    policy: GatePolicyRequest = Field(default_factory=GatePolicyRequest)
    label_rules: list[LabelRuleRequest] = Field(default_factory=list)

    def to_rules(self) -> tuple[LabelConsistencyRule, ...]:
        return tuple(r.to_domain() for r in self.label_rules)


class RemediationRequest(BaseModel):
    kind: Literal[
        "DROP_ROWS",
        "IMPUTE",
        "CLIP",
        "RELABEL",
        "DEDUPLICATE",
        "SMOOTH",
        "RESAMPLE",
        "EXCLUDE_SEGMENT",
        "RECOLLECT",
    ]
    dimension: Literal[
        "COMPLETENESS",
        "VALIDITY",
        "LABEL_QUALITY",
        "BALANCE",
        "NOISE",
        "UNIQUENESS",
    ]
    target: str
    affected_rows: int = Field(ge=0)
    rationale: str = Field(min_length=5)
    decided_by: str = Field(min_length=1)

    def to_domain(self) -> RemediationAction:
        return RemediationAction(
            kind=RemediationKind(self.kind),
            dimension=QualityDimension(self.dimension),
            target=self.target,
            affected_rows=self.affected_rows,
            rationale=self.rationale,
            decided_by=self.decided_by,
        )


class CompareQualityRequest(BaseModel):
    before_assessment_id: str
    after_assessment_id: str
    before_label: str = "before"
    after_label: str = "after"
    policy: GatePolicyRequest = Field(default_factory=GatePolicyRequest)


class ReopenAssessmentRequest(BaseModel):
    reason: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# 응답
# ---------------------------------------------------------------------------
class DimensionResponse(BaseModel):
    assessment_id: str
    dimension: str
    score: float
    grade: str
    verdict: str
    findings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: DimensionView) -> DimensionResponse:
        return cls(
            assessment_id=view.assessment_id,
            dimension=view.dimension,
            score=view.score,
            grade=view.grade,
            verdict=view.verdict,
            findings=[FindingResponse.from_view(f) for f in view.findings],
        )


class TrainingImpactResponse(BaseModel):
    total_rows: int
    usable_rows: int
    distinct_rows: int
    rows_with_missing: int
    conflicting_rows: int
    inflation_ratio: float
    baseline_accuracy: float
    accuracy_ceiling: float
    minority_count: int

    @classmethod
    def from_view(cls, view: TrainingImpactView) -> TrainingImpactResponse:
        return cls(
            total_rows=view.total_rows,
            usable_rows=view.usable_rows,
            distinct_rows=view.distinct_rows,
            rows_with_missing=view.rows_with_missing,
            conflicting_rows=view.conflicting_rows,
            inflation_ratio=view.inflation_ratio,
            baseline_accuracy=view.baseline_accuracy,
            accuracy_ceiling=view.accuracy_ceiling,
            minority_count=view.minority_count,
        )


class QualityScoreResponse(BaseModel):
    assessment_id: str
    dataset_ref: str
    overall_score: float
    grade: str
    dimensions: list[DimensionResponse]
    impact: TrainingImpactResponse | None = None

    @classmethod
    def from_view(cls, view: QualityScoreView) -> QualityScoreResponse:
        return cls(
            assessment_id=view.assessment_id,
            dataset_ref=view.dataset_ref,
            overall_score=view.overall_score,
            grade=view.grade,
            dimensions=[DimensionResponse.from_view(d) for d in view.dimensions],
            impact=(
                TrainingImpactResponse.from_view(view.impact) if view.impact else None
            ),
        )


class QualityGateResponse(BaseModel):
    assessment_id: str
    dataset_ref: str
    verdict: str
    is_ready: bool
    overall_score: float
    grade: str
    dimension_scores: list[dict[str, float | str]]
    missing_dimensions: list[str]
    blocking_reasons: list[str]
    blocking: list[FindingResponse]
    warnings: list[FindingResponse]

    @classmethod
    def from_view(cls, view: QualityGateView) -> QualityGateResponse:
        return cls(
            assessment_id=view.assessment_id,
            dataset_ref=view.dataset_ref,
            verdict=view.verdict,
            is_ready=view.is_ready,
            overall_score=view.overall_score,
            grade=view.grade,
            dimension_scores=[
                {"dimension": name, "score": score, "verdict": verdict}
                for name, score, verdict in view.dimension_scores
            ],
            missing_dimensions=list(view.missing_dimensions),
            blocking_reasons=list(view.blocking_reasons),
            blocking=[FindingResponse.from_view(f) for f in view.blocking],
            warnings=[FindingResponse.from_view(f) for f in view.warnings],
        )


class AssessmentResponse(BaseModel):
    assessment_id: str
    dataset_ref: str
    status: str
    measured_dimensions: list[str]
    unverified_dimensions: list[str]
    remediations: list[str]
    verdict: str | None = None

    @classmethod
    def from_view(cls, view: AssessmentView) -> AssessmentResponse:
        return cls(
            assessment_id=view.assessment_id,
            dataset_ref=view.dataset_ref,
            status=view.status,
            measured_dimensions=list(view.measured_dimensions),
            unverified_dimensions=list(view.unverified_dimensions),
            remediations=list(view.remediations),
            verdict=view.verdict,
        )


class QualityComparisonResponse(BaseModel):
    before_label: str
    after_label: str
    before_overall: float
    after_overall: float
    overall_delta: float
    dimensions: list[dict[str, float | str | None]]
    improved: list[str]
    regressed: list[str]
    before_impact: TrainingImpactResponse | None = None
    after_impact: TrainingImpactResponse | None = None

    @classmethod
    def from_view(cls, view: QualityComparisonView) -> QualityComparisonResponse:
        return cls(
            before_label=view.before_label,
            after_label=view.after_label,
            before_overall=view.before_overall,
            after_overall=view.after_overall,
            overall_delta=view.overall_delta,
            dimensions=[
                {
                    "dimension": name,
                    "before": before,
                    "after": after,
                    "delta": delta,
                }
                for name, before, after, delta in view.deltas
            ],
            improved=list(view.improved),
            regressed=list(view.regressed),
            before_impact=(
                TrainingImpactResponse.from_view(view.before_impact)
                if view.before_impact
                else None
            ),
            after_impact=(
                TrainingImpactResponse.from_view(view.after_impact)
                if view.after_impact
                else None
            ),
        )
