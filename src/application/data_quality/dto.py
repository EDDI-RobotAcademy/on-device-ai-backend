"""Data Quality Use Case 결과 DTO."""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from domain.data_quality.assessment import QualityAssessment
from domain.data_quality.comparison import QualityComparison, TrainingImpact
from domain.data_quality.dimensions import DimensionResult
from domain.data_quality.gate import QualityCertificate
from domain.data_quality.rebalancing import RebalancingComparison


@dataclass(frozen=True, slots=True)
class DimensionView:
    assessment_id: str
    dimension: str
    score: float
    grade: str
    verdict: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, assessment_id: str, result: DimensionResult) -> DimensionView:
        return cls(
            assessment_id=assessment_id,
            dimension=result.dimension.value,
            score=result.score.value,
            grade=result.score.grade.value,
            verdict=result.verdict.value,
            findings=tuple(FindingView.of(f) for f in result.findings),
        )

    @property
    def passed(self) -> bool:
        return self.verdict != "FAILED"

    def render(self) -> str:
        lines = [f"[{self.dimension}] {self.score:.1f} ({self.grade})  {self.verdict}"]
        lines += [f"  - {f.describe()}" for f in self.findings]
        if not self.findings:
            lines.append("  - (지적 사항 없음)")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TrainingImpactView:
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
    def of(cls, impact: TrainingImpact) -> TrainingImpactView:
        return cls(
            total_rows=impact.total_rows,
            usable_rows=impact.usable_rows,
            distinct_rows=impact.distinct_rows,
            rows_with_missing=impact.rows_with_missing,
            conflicting_rows=impact.conflicting_rows,
            inflation_ratio=impact.inflation_ratio,
            baseline_accuracy=impact.baseline_accuracy,
            accuracy_ceiling=impact.accuracy_ceiling,
            minority_count=impact.minority_count,
        )

    def render(self) -> str:
        return (
            f"전체 {self.total_rows:,}행 → 학습 가능 {self.usable_rows:,}행 "
            f"(부풀림 {self.inflation_ratio:.2f}배)\n"
            f"  결측 포함 행 {self.rows_with_missing:,} / 라벨 모순 {self.conflicting_rows:,}\n"
            f"  정확도 상한 {self.accuracy_ceiling:.2%} / "
            f"baseline 정확도 {self.baseline_accuracy:.2%} / "
            f"소수 클래스 {self.minority_count:,}개"
        )


@dataclass(frozen=True, slots=True)
class QualityScoreView:
    assessment_id: str
    dataset_ref: str
    overall_score: float
    grade: str
    dimensions: tuple[DimensionView, ...]
    impact: TrainingImpactView | None = None

    def render(self) -> str:
        lines = [
            f"품질 종합 ({self.dataset_ref}) — {self.overall_score:.1f} ({self.grade})",
            "",
            f"{'차원':<16}{'점수':>8}{'등급':>6}  판정",
            "-" * 50,
        ]
        for view in self.dimensions:
            lines.append(
                f"{view.dimension:<16}{view.score:>8.1f}{view.grade:>6}  {view.verdict}"
            )
        if self.impact:
            lines.append("")
            lines.append("학습 관점")
            lines.append("  " + self.impact.render().replace("\n", "\n  "))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class QualityGateView:
    assessment_id: str
    dataset_ref: str
    verdict: str
    is_ready: bool
    overall_score: float
    grade: str
    dimension_scores: tuple[tuple[str, float, str], ...]
    missing_dimensions: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    blocking: tuple[FindingView, ...]
    warnings: tuple[FindingView, ...]

    @classmethod
    def of(cls, assessment_id: str, certificate: QualityCertificate) -> QualityGateView:
        return cls(
            assessment_id=assessment_id,
            dataset_ref=certificate.dataset_ref,
            verdict=certificate.verdict.value,
            is_ready=certificate.is_ready,
            overall_score=certificate.overall_score.value,
            grade=certificate.grade.value,
            dimension_scores=tuple(
                (d.value, score, verdict)
                for d, score, verdict in certificate.dimension_scores
            ),
            missing_dimensions=tuple(
                d.value for d in certificate.missing_dimensions
            ),
            blocking_reasons=certificate.blocking_reasons,
            blocking=tuple(FindingView.of(f) for f in certificate.blocking_findings),
            warnings=tuple(FindingView.of(f) for f in certificate.warning_findings),
        )

    def render(self) -> str:
        lines = [
            f"Data Quality Gate: {self.verdict}  ({self.dataset_ref})",
            f"종합 점수: {self.overall_score:.1f} ({self.grade})",
            "",
            f"{'차원':<16}{'점수':>8}  판정",
            "-" * 44,
        ]
        for name, score, verdict in self.dimension_scores:
            lines.append(f"{name:<16}{score:>8.1f}  {verdict}")
        if self.missing_dimensions:
            lines.append("")
            lines.append("측정하지 않은 축: " + ", ".join(self.missing_dimensions))
        if self.blocking_reasons:
            lines.append("")
            lines.append("차단 사유:")
            lines += [f"  ✗ {reason}" for reason in self.blocking_reasons]
        if self.warnings:
            lines.append("")
            lines.append("경고(진행은 가능):")
            lines += [f"  ! {f.describe()}" for f in self.warnings]
        if self.is_ready and not self.warnings:
            lines.append("")
            lines.append("  ✓ 막는 것도, 걸리는 것도 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class AssessmentView:
    assessment_id: str
    dataset_ref: str
    status: str
    measured_dimensions: tuple[str, ...]
    unverified_dimensions: tuple[str, ...]
    remediations: tuple[str, ...]
    verdict: str | None

    @classmethod
    def of(cls, assessment: QualityAssessment) -> AssessmentView:
        return cls(
            assessment_id=str(assessment.id),
            dataset_ref=assessment.dataset_ref,
            status=assessment.status.value,
            measured_dimensions=tuple(
                d.value for d in assessment.measured_dimensions
            ),
            unverified_dimensions=tuple(
                sorted(d.value for d in assessment.unverified_dimensions)
            ),
            remediations=tuple(a.describe() for a in assessment.remediations),
            verdict=(
                assessment.latest_verdict.value if assessment.latest_verdict else None
            ),
        )


@dataclass(frozen=True, slots=True)
class QualityComparisonView:
    before_label: str
    after_label: str
    before_overall: float
    after_overall: float
    overall_delta: float
    deltas: tuple[tuple[str, float | None, float | None, float | None], ...]
    improved: tuple[str, ...]
    regressed: tuple[str, ...]
    before_impact: TrainingImpactView | None
    after_impact: TrainingImpactView | None
    _rendered: str = ""

    @classmethod
    def of(cls, comparison: QualityComparison) -> QualityComparisonView:
        dimensions = sorted(
            set(comparison.before.dimension_scores)
            | set(comparison.after.dimension_scores),
            key=lambda d: d.value,
        )
        return cls(
            before_label=comparison.before.label,
            after_label=comparison.after.label,
            before_overall=comparison.before.overall_score.value,
            after_overall=comparison.after.overall_score.value,
            overall_delta=comparison.overall_delta,
            deltas=tuple(
                (
                    d.value,
                    comparison.before.dimension_scores.get(d),
                    comparison.after.dimension_scores.get(d),
                    comparison.delta_of(d),
                )
                for d in dimensions
            ),
            improved=tuple(d.value for d in comparison.improved),
            regressed=tuple(d.value for d in comparison.regressed),
            before_impact=(
                TrainingImpactView.of(comparison.before.impact)
                if comparison.before.impact
                else None
            ),
            after_impact=(
                TrainingImpactView.of(comparison.after.impact)
                if comparison.after.impact
                else None
            ),
            _rendered=comparison.render(),
        )

    def render(self) -> str:
        return self._rendered


@dataclass(frozen=True, slots=True)
class RebalancingOutcomeView:
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


@dataclass(frozen=True, slots=True)
class RebalancingComparisonView:
    """불균형 완화 전략 비교. (실습 2-11)"""

    dataset_id: str
    outcomes: tuple[RebalancingOutcomeView, ...]
    safe_strategies: tuple[str, ...]
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        dataset_id: str,
        comparison: RebalancingComparison,
        findings: tuple[FindingView, ...] = (),
    ) -> RebalancingComparisonView:
        return cls(
            dataset_id=dataset_id,
            outcomes=tuple(
                RebalancingOutcomeView(
                    strategy=plan.strategy.value,
                    plan=plan.describe(),
                    total_before=outcome.total_before,
                    total_after=outcome.total_after,
                    imbalance_before=outcome.imbalance_before,
                    imbalance_after=outcome.imbalance_after,
                    duplicated_rows=outcome.duplicated_rows,
                    discarded_rows=outcome.discarded_rows,
                    synthesized_rows=outcome.synthesized_rows,
                    distinct_minority_samples=outcome.distinct_minority_samples,
                    information_gain=outcome.information_gain,
                    verdict=(
                        "BLOCKED"
                        if any(f.is_blocking for f in row_findings)
                        else ("WARNED" if row_findings else "OK")
                    ),
                )
                for plan, outcome, row_findings in comparison.rows
            ),
            safe_strategies=tuple(s.value for s in comparison.safe_strategies),
            table=comparison.render(),
            findings=findings,
        )

    def outcome_of(self, strategy: str) -> RebalancingOutcomeView | None:
        return next((o for o in self.outcomes if o.strategy == strategy), None)

    def render(self) -> str:
        return f"불균형 완화 비교 ({self.dataset_id})\n{self.table}"
