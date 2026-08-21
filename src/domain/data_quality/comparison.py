"""망가진 데이터와 정상 데이터의 비교. (실습 2-9)

"품질이 좋아졌다"는 말은 근거가 아니다. 근거는 다음 형태여야 한다.

    학습 가능 표본     8,640 → 7,912   (중복·모순 제거)
    정확도 상한        96.8% → 99.7%   (라벨 오류 정정)
    baseline 정확도    99.4% → 92.1%   (재표집)
    COMPLETENESS       42.3 → 96.0

점수만 비교하면 "숫자가 올랐다"로 끝난다.
학습 관점의 영향까지 함께 봐야 무엇을 얻었는지 알 수 있다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.data_quality.balance import ClassBalanceMeasurement
from domain.data_quality.completeness import MissingValueMeasurement
from domain.data_quality.dimensions import (
    DimensionResult,
    QualityDimension,
    QualityScore,
)
from domain.data_quality.label_quality import LabelErrorMeasurement
from domain.data_quality.uniqueness import DuplicateMeasurement
from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class TrainingImpact:
    """품질이 학습에 미치는 영향을 표본 수로 환산한 값."""

    total_rows: int
    distinct_rows: int
    rows_with_missing: int
    conflicting_rows: int
    baseline_accuracy: float
    minority_count: int
    accuracy_ceiling: float

    def __post_init__(self) -> None:
        if self.total_rows < 0:
            raise InvariantViolation("total_rows 는 음수일 수 없다.", subject="total_rows")

    @property
    def usable_rows(self) -> int:
        """중복과 모순을 걷어낸 뒤 실제로 학습에 쓸 수 있는 표본 수의 근사."""
        return max(self.distinct_rows - self.conflicting_rows, 0)

    @property
    def inflation_ratio(self) -> float:
        if self.usable_rows == 0:
            return float("inf")
        return self.total_rows / self.usable_rows

    def render(self) -> str:
        return (
            f"전체 {self.total_rows:,}행 → 학습 가능 {self.usable_rows:,}행 "
            f"(부풀림 {self.inflation_ratio:.2f}배)\n"
            f"  정확도 상한 {self.accuracy_ceiling:.2%} / "
            f"baseline 정확도 {self.baseline_accuracy:.2%} / "
            f"소수 클래스 {self.minority_count:,}개"
        )


def estimate_training_impact(
    *,
    missing: MissingValueMeasurement,
    duplicates: DuplicateMeasurement,
    balance: ClassBalanceMeasurement,
    labels: LabelErrorMeasurement,
) -> TrainingImpact:
    """네 축의 측정값을 '표본 수'라는 하나의 언어로 환산한다.

    이것은 Domain Service 다. 어떤 측정기가 값을 냈는지는 알 필요가 없다.
    """
    return TrainingImpact(
        total_rows=duplicates.total_rows,
        distinct_rows=duplicates.distinct_row_count,
        rows_with_missing=missing.rows_with_any_missing_upper_bound,
        conflicting_rows=duplicates.conflicting_label_count,
        baseline_accuracy=balance.baseline_accuracy,
        minority_count=balance.minority_count,
        accuracy_ceiling=labels.accuracy_ceiling(),
    )


@dataclass(frozen=True, slots=True)
class QualitySnapshot:
    """한 시점의 품질 상태."""

    label: str
    overall_score: QualityScore
    dimension_scores: Mapping[QualityDimension, float] = field(default_factory=dict)
    impact: TrainingImpact | None = None

    @classmethod
    def of(
        cls,
        label: str,
        overall: QualityScore,
        results: Mapping[QualityDimension, DimensionResult],
        impact: TrainingImpact | None = None,
    ) -> QualitySnapshot:
        return cls(
            label=label,
            overall_score=overall,
            dimension_scores={d: r.score.value for d, r in results.items()},
            impact=impact,
        )


@dataclass(frozen=True, slots=True)
class QualityComparison:
    before: QualitySnapshot
    after: QualitySnapshot

    @property
    def overall_delta(self) -> float:
        return self.after.overall_score.value - self.before.overall_score.value

    def delta_of(self, dimension: QualityDimension) -> float | None:
        before = self.before.dimension_scores.get(dimension)
        after = self.after.dimension_scores.get(dimension)
        if before is None or after is None:
            return None
        return after - before

    @property
    def improved(self) -> tuple[QualityDimension, ...]:
        return tuple(
            d
            for d in sorted(self.after.dimension_scores, key=lambda d: d.value)
            if (self.delta_of(d) or 0.0) > 0.5
        )

    @property
    def regressed(self) -> tuple[QualityDimension, ...]:
        return tuple(
            d
            for d in sorted(self.after.dimension_scores, key=lambda d: d.value)
            if (self.delta_of(d) or 0.0) < -0.5
        )

    def render(self) -> str:
        lines = [
            f"품질 비교: {self.before.label} → {self.after.label}",
            "",
            f"{'차원':<16}{'before':>9}{'after':>9}{'delta':>9}",
            "-" * 43,
        ]
        for dimension in sorted(
            set(self.before.dimension_scores) | set(self.after.dimension_scores),
            key=lambda d: d.value,
        ):
            before = self.before.dimension_scores.get(dimension)
            after = self.after.dimension_scores.get(dimension)
            delta = self.delta_of(dimension)
            lines.append(
                f"{dimension.value:<16}"
                f"{'-' if before is None else f'{before:>9.1f}'}"
                f"{'-' if after is None else f'{after:>9.1f}'}"
                f"{'-' if delta is None else f'{delta:>+9.1f}'}"
            )
        lines.append("-" * 43)
        lines.append(
            f"{'종합':<16}{self.before.overall_score.value:>9.1f}"
            f"{self.after.overall_score.value:>9.1f}{self.overall_delta:>+9.1f}"
        )
        if self.before.impact and self.after.impact:
            lines.append("")
            lines.append("학습 관점")
            lines.append(f"  before : {self.before.impact.render()}")
            lines.append(f"  after  : {self.after.impact.render()}")
        if self.regressed:
            lines.append("")
            lines.append(
                "나빠진 축: " + ", ".join(d.value for d in self.regressed)
            )
        return "\n".join(lines)
