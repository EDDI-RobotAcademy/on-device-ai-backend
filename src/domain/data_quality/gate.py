"""Data Quality Gate — 통과시킬 것인가. (실습 2-8, 2-10)

**점수와 게이트는 다르다.**

    점수(Score)   대화를 위한 것이다. "COMPLETENESS 42점" 은 회의를 바꾼다.
    게이트(Gate)  결정을 위한 것이다. 통과 아니면 차단, 그 사이는 없다.

모듈 1의 판정은 순수하게 심각도 기반이었다 (CRITICAL 하나면 차단).
품질은 그렇게 다룰 수 없다. 잡음이 조금 있는 데이터는 흔하고, 그때마다 막으면
아무도 이 게이트를 쓰지 않는다. 그래서 여기서는 **점수 + 차단 축**을 함께 쓴다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.data_quality.balance import BalancePolicy
from domain.data_quality.completeness import CompletenessPolicy
from domain.data_quality.dimensions import (
    ALL_DIMENSIONS,
    DimensionResult,
    QualityDimension,
    QualityGrade,
    QualityScore,
)
from domain.data_quality.label_quality import LabelConsistencyRule, LabelQualityPolicy
from domain.data_quality.noise import NoisePolicy
from domain.data_quality.uniqueness import UniquenessPolicy
from domain.data_quality.validity import ValidityPolicy
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict

DEFAULT_WEIGHTS: Mapping[QualityDimension, float] = {
    QualityDimension.COMPLETENESS: 0.20,
    QualityDimension.VALIDITY: 0.20,
    QualityDimension.LABEL_QUALITY: 0.25,
    QualityDimension.BALANCE: 0.10,
    QualityDimension.NOISE: 0.10,
    QualityDimension.UNIQUENESS: 0.15,
}
"""라벨 품질이 가장 무겁다. 다른 모든 축은 고칠 수 있지만,
정답이 틀린 데이터는 아무리 고쳐도 그 이상 좋아지지 않기 때문이다."""


@dataclass(frozen=True, slots=True)
class QualityRuleSet:
    """여섯 축의 기준을 한 묶음으로 들고 다닌다.

    라인마다, 제품마다 다를 수 있다. 그래서 Value Object 다.
    """

    completeness: CompletenessPolicy = field(default_factory=CompletenessPolicy)
    validity: ValidityPolicy = field(default_factory=ValidityPolicy)
    label_quality: LabelQualityPolicy = field(default_factory=LabelQualityPolicy)
    balance: BalancePolicy = field(default_factory=BalancePolicy)
    noise: NoisePolicy = field(default_factory=NoisePolicy)
    uniqueness: UniquenessPolicy = field(default_factory=UniquenessPolicy)
    label_rules: tuple[LabelConsistencyRule, ...] = field(default_factory=tuple)
    """라벨 일관성 규칙. 현장에서 받아 적는다. 비어 있으면 라벨 검사가 차단된다."""


@dataclass(frozen=True, slots=True)
class QualityCertificate:
    """게이트 판정의 결과이자 근거 기록."""

    dataset_ref: str
    verdict: Verdict
    overall_score: QualityScore
    dimension_scores: tuple[tuple[QualityDimension, float, str], ...]
    missing_dimensions: tuple[QualityDimension, ...]
    blocking_reasons: tuple[str, ...]
    blocking_findings: tuple[Finding, ...]
    warning_findings: tuple[Finding, ...]

    @property
    def is_ready(self) -> bool:
        return self.verdict is not Verdict.FAILED

    @property
    def grade(self) -> QualityGrade:
        return self.overall_score.grade

    def render(self) -> str:
        lines = [
            f"Data Quality Gate: {self.verdict.value}  ({self.dataset_ref})",
            f"종합 점수: {self.overall_score}",
            "",
            f"{'차원':<16}{'점수':>8}  판정",
            "-" * 44,
        ]
        for dimension, score, verdict in self.dimension_scores:
            lines.append(f"{dimension.value:<16}{score:>8.1f}  {verdict}")
        if self.missing_dimensions:
            lines.append("")
            lines.append(
                "측정하지 않은 축: "
                + ", ".join(d.value for d in self.missing_dimensions)
            )
        if self.blocking_reasons:
            lines.append("")
            lines.append("차단 사유:")
            lines += [f"  ✗ {reason}" for reason in self.blocking_reasons]
        if self.warning_findings:
            lines.append("")
            lines.append(f"경고 {len(self.warning_findings)}건 (진행은 가능)")
        if self.is_ready and not self.warning_findings:
            lines.append("")
            lines.append("  ✓ 막는 것도, 걸리는 것도 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class QualityGatePolicy:
    """무엇을 통과라고 부를 것인가."""

    weights: Mapping[QualityDimension, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    minimum_overall_score: float = 80.0
    minimum_dimension_score: float = 50.0
    """한 축이라도 이 아래로 떨어지면, 종합 점수가 높아도 통과시키지 않는다."""

    required_dimensions: frozenset[QualityDimension] = field(
        default_factory=lambda: frozenset(ALL_DIMENSIONS)
    )
    blocking_dimensions: frozenset[QualityDimension] = field(
        default_factory=lambda: frozenset(
            {QualityDimension.LABEL_QUALITY, QualityDimension.COMPLETENESS}
        )
    )
    """이 축이 FAILED 면 점수와 무관하게 차단한다."""

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise InvariantViolation(
                f"가중치의 합이 {total:.4f} 다. 1.0 이어야 한다.", subject="weights"
            )
        if any(w < 0 for w in self.weights.values()):
            raise InvariantViolation("가중치는 음수일 수 없다.", subject="weights")
        if not 0.0 <= self.minimum_overall_score <= 100.0:
            raise InvariantViolation(
                "minimum_overall_score 는 0~100 이어야 한다.",
                subject="minimum_overall_score",
            )
        missing = set(self.required_dimensions) - set(self.weights)
        if missing:
            raise InvariantViolation(
                f"가중치가 없는 필수 축이 있다: {sorted(d.value for d in missing)}",
                subject="weights",
            )

    def overall_score(
        self, results: Mapping[QualityDimension, DimensionResult]
    ) -> QualityScore:
        """측정된 축들의 가중 평균.

        측정하지 않은 축은 0점으로 치지 않는다. 그것은 별도의 차단 사유다.
        점수를 0으로 만들면 '왜 낮은지'가 뭉개진다.
        """
        weighted = 0.0
        total_weight = 0.0
        for dimension, result in results.items():
            weight = self.weights.get(dimension, 0.0)
            weighted += weight * result.score.value
            total_weight += weight
        if total_weight == 0:
            return QualityScore(0.0)
        return QualityScore(weighted / total_weight)

    def evaluate(
        self,
        dataset_ref: str,
        results: Mapping[QualityDimension, DimensionResult],
        *,
        unverified_dimensions: frozenset[QualityDimension] = frozenset(),
    ) -> QualityCertificate:
        missing = tuple(
            sorted(
                (d for d in self.required_dimensions if d not in results),
                key=lambda d: d.value,
            )
        )
        overall = self.overall_score(results)

        reasons: list[str] = []
        blocking: list[Finding] = []
        warnings: list[Finding] = []

        for dimension in sorted(results, key=lambda d: d.value):
            result = results[dimension]
            blocking.extend(result.blocking)
            warnings.extend(result.warnings)

            if result.score.value < self.minimum_dimension_score:
                reasons.append(
                    f"{dimension.value} 점수 {result.score.value:.1f} < "
                    f"{self.minimum_dimension_score:.0f}"
                )
            if dimension in self.blocking_dimensions and not result.passed:
                reasons.append(
                    f"{dimension.value} 는 차단 축이며 판정이 {result.verdict.value} 다"
                )

        for dimension in missing:
            reasons.append(f"{dimension.value} 를 측정하지 않았다")
            blocking.append(
                Finding(
                    code="GATE_DIMENSION_NOT_MEASURED",
                    message=f"{dimension.value} 축을 측정하지 않았다.",
                    severity=Severity.CRITICAL,
                    subject=dimension.value,
                )
            )

        for dimension in sorted(unverified_dimensions, key=lambda d: d.value):
            reasons.append(
                f"{dimension.value} 에 조치를 기록했으나 재측정하지 않았다"
            )
            blocking.append(
                Finding(
                    code="GATE_REMEDIATION_UNVERIFIED",
                    message=(
                        "데이터를 고쳤다고 기록했지만 다시 측정하지 않았다. "
                        "고쳤다는 주장은 측정으로만 확인된다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=dimension.value,
                )
            )

        if overall.value < self.minimum_overall_score:
            reasons.append(
                f"종합 점수 {overall.value:.1f} < {self.minimum_overall_score:.0f}"
            )

        verdict = (
            Verdict.FAILED
            if reasons
            else (Verdict.PASSED_WITH_WARNINGS if warnings else Verdict.PASSED)
        )

        return QualityCertificate(
            dataset_ref=dataset_ref,
            verdict=verdict,
            overall_score=overall,
            dimension_scores=tuple(
                (d, results[d].score.value, results[d].verdict.value)
                for d in sorted(results, key=lambda d: d.value)
            ),
            missing_dimensions=missing,
            blocking_reasons=tuple(reasons),
            blocking_findings=tuple(blocking),
            warning_findings=tuple(warnings),
        )
