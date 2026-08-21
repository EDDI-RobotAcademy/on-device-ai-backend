"""라벨 오류 — 정답이 틀렸다. (실습 2-4)

모듈 1(실습 1-6)에서는 "라벨의 **정의**가 있는가, 사람들이 **합의**했는가"를 물었다.
여기서는 다른 것을 묻는다. **붙여 놓은 라벨이 실제 데이터와 모순되지 않는가.**

라벨 오류는 데이터만 봐서는 찾을 수 없다. 현장 규칙이 있어야 찾을 수 있다.

    "FAULT 라면 유효전력이 정격 아래로 떨어져 있어야 한다"
    → 그런데 FAULT 인데 전력이 정상 범위인 행이 60건 있다
    → 둘 중 하나다. 라벨이 틀렸거나, FAULT 의 정의가 틀렸다.

그리고 가장 치명적인 것: **같은 입력에 다른 라벨.**
모델은 이 모순을 절대 학습할 수 없다. 정확도 상한이 그 자리에서 깎인다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.data_quality.dimensions import (
    DimensionResult,
    QualityDimension,
    QualityScore,
    deduct,
)
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class LabelConsistencyRule:
    """라벨이 성립하려면 데이터가 만족해야 하는 조건.

    이 규칙은 데이터에서 나오지 않는다. 설비 담당자에게서 나온다.
    """

    label: str
    field_name: str
    expected_min: float | None = None
    expected_max: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise InvariantViolation("규칙이 어느 라벨에 대한 것인지 없다.", subject="label")
        if self.expected_min is None and self.expected_max is None:
            raise InvariantViolation(
                f"'{self.label}' 규칙에 조건이 없다. 조건 없는 규칙은 아무것도 검증하지 못한다.",
                subject=self.label,
            )
        if (
            self.expected_min is not None
            and self.expected_max is not None
            and self.expected_min > self.expected_max
        ):
            raise InvariantViolation("조건 범위가 뒤집혀 있다.", subject=self.label)
        if not self.description.strip():
            raise InvariantViolation(
                f"'{self.label}' 규칙에 근거 설명이 없다. 나중에 아무도 이 숫자를 못 고친다.",
                subject=self.label,
            )

    def describe(self) -> str:
        if self.expected_min is not None and self.expected_max is not None:
            condition = f"{self.expected_min:g} ~ {self.expected_max:g}"
        elif self.expected_min is not None:
            condition = f"{self.expected_min:g} 이상"
        else:
            condition = f"{self.expected_max:g} 이하"
        return f"{self.label} → {self.field_name} 이 {condition} ({self.description})"


@dataclass(frozen=True, slots=True)
class LabelErrorMeasurement:
    total_labeled: int
    rule_violations: Mapping[str, int] = field(default_factory=dict)
    """규칙 설명 → 위반 표본 수."""

    violations_by_label: Mapping[str, int] = field(default_factory=dict)
    conflicting_duplicate_count: int = 0
    """입력이 같은데 라벨이 다른 표본 수. 모델이 절대 학습할 수 없는 모순이다."""

    conflicting_group_count: int = 0
    examples: tuple[str, ...] = field(default_factory=tuple)
    """사람이 눈으로 확인할 대표 사례."""

    def __post_init__(self) -> None:
        if self.total_labeled < 0:
            raise InvariantViolation("total_labeled 는 음수일 수 없다.", subject="total_labeled")
        if self.conflicting_duplicate_count > self.total_labeled:
            raise InvariantViolation(
                "모순 표본 수가 전체보다 클 수 없다.", subject="conflicting_duplicate_count"
            )

    @property
    def total_violation_count(self) -> int:
        return sum(self.rule_violations.values())

    @property
    def violation_ratio(self) -> float:
        if self.total_labeled == 0:
            return 0.0
        return self.total_violation_count / self.total_labeled

    @property
    def conflict_ratio(self) -> float:
        if self.total_labeled == 0:
            return 0.0
        return self.conflicting_duplicate_count / self.total_labeled

    def accuracy_ceiling(self) -> float:
        """라벨이 틀린 만큼은 아무리 좋은 모델도 맞힐 수 없다.

        정확도 상한의 아주 거친 근사다. 정확한 값을 주려는 것이 아니라,
        "라벨 오류 2% 는 정확도 2% 를 그냥 버리는 것"이라는 감각을 주기 위한 것이다.
        """
        return max(0.0, 1.0 - self.violation_ratio - self.conflict_ratio)


@dataclass(frozen=True, slots=True)
class LabelQualityPolicy:
    max_violation_ratio: float = 0.005
    max_conflict_ratio: float = 0.0
    """같은 입력에 다른 라벨은 단 한 건도 허용하지 않는다."""

    min_accuracy_ceiling: float = 0.98

    def __post_init__(self) -> None:
        for name in ("max_violation_ratio", "max_conflict_ratio", "min_accuracy_ceiling"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    def evaluate(
        self,
        measurement: LabelErrorMeasurement,
        rules: tuple[LabelConsistencyRule, ...] = (),
    ) -> DimensionResult:
        findings: list[Finding] = []
        deductions: list[float] = []

        if not rules:
            findings.append(
                Finding(
                    code="LABEL_NO_CONSISTENCY_RULE",
                    message=(
                        "라벨 일관성 규칙이 하나도 없다. 규칙이 없으면 라벨 오류를 "
                        "찾을 방법 자체가 없고, '오류 0건'은 근거가 아니다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="rules",
                    measured=0.0,
                    threshold=1.0,
                )
            )
            deductions.append(40.0)

        if measurement.violation_ratio > self.max_violation_ratio:
            findings.append(
                Finding(
                    code="LABEL_RULE_VIOLATION",
                    message=(
                        f"현장 규칙과 모순되는 라벨이 {measurement.total_violation_count} 건이다. "
                        "라벨이 틀렸거나, 규칙이 틀렸다. 둘 다 사람이 확인해야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="label",
                    measured=measurement.violation_ratio,
                    threshold=self.max_violation_ratio,
                )
            )
            deductions.append(
                deduct(
                    measurement.violation_ratio,
                    tolerance=self.max_violation_ratio,
                    cap=0.05,
                    weight=40.0,
                )
            )

        for label, count in sorted(measurement.violations_by_label.items()):
            if count == 0:
                continue
            findings.append(
                Finding(
                    code="LABEL_VIOLATION_BY_CLASS",
                    message=f"'{label}' 라벨에서 규칙 위반 {count} 건.",
                    severity=Severity.INFO,
                    subject=label,
                    measured=float(count),
                )
            )

        if measurement.conflict_ratio > self.max_conflict_ratio:
            findings.append(
                Finding(
                    code="LABEL_CONFLICT",
                    message=(
                        f"입력이 같은데 라벨이 다른 표본이 {measurement.conflicting_duplicate_count} 건 "
                        f"({measurement.conflicting_group_count} 개 묶음)이다. "
                        "모델은 이 모순을 학습할 수 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="label",
                    measured=measurement.conflict_ratio,
                    threshold=self.max_conflict_ratio,
                )
            )
            deductions.append(
                deduct(
                    measurement.conflict_ratio,
                    tolerance=self.max_conflict_ratio,
                    cap=0.02,
                    weight=40.0,
                )
            )

        ceiling = measurement.accuracy_ceiling()
        if ceiling < self.min_accuracy_ceiling:
            findings.append(
                Finding(
                    code="LABEL_ACCURACY_CEILING",
                    message=(
                        f"라벨 오류만으로 정확도 상한이 {ceiling:.1%} 로 내려간다. "
                        "모델을 아무리 키워도 이 선을 넘을 수 없다."
                    ),
                    severity=Severity.WARNING,
                    subject="label",
                    measured=ceiling,
                    threshold=self.min_accuracy_ceiling,
                )
            )

        return DimensionResult(
            dimension=QualityDimension.LABEL_QUALITY,
            score=QualityScore.from_deductions(deductions),
            findings=tuple(findings),
        )
