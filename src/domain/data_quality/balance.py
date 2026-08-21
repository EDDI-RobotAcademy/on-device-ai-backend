"""불균형 — 한쪽으로 쏠렸다. (실습 2-5)

불량률 0.3% 인 라인에서 "정확도 99.7%" 는 아무 의미가 없다.
전부 정상이라고 찍어도 나오는 숫자다.

불균형은 없앨 문제가 아니라 **알고 대응할 문제**다. 그래서 여기서 재는 것은
"얼마나 쏠렸는가"가 아니라 **"쏠림 때문에 무엇을 못 하게 되는가"**다.

    baseline_accuracy       다수 클래스만 찍었을 때의 정확도
    minority_count          소수 클래스의 절대 표본 수 (비율이 아니라 개수가 중요하다)
    effective_sample_count  분할 후 test 에 남을 소수 클래스 표본 수
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
class ClassBalanceMeasurement:
    class_counts: Mapping[str, int] = field(default_factory=dict)
    test_split_ratio: float = 0.15
    """분할 후 평가에 쓰일 비율. 소수 클래스가 test 에 몇 개나 남는지 보기 위한 값."""

    def __post_init__(self) -> None:
        if any(count < 0 for count in self.class_counts.values()):
            raise InvariantViolation("클래스 개수는 음수일 수 없다.", subject="class_counts")
        if not 0.0 < self.test_split_ratio < 1.0:
            raise InvariantViolation(
                "test_split_ratio 는 0과 1 사이여야 한다.", subject="test_split_ratio"
            )

    @property
    def total_count(self) -> int:
        return sum(self.class_counts.values())

    @property
    def class_count(self) -> int:
        return len([c for c in self.class_counts.values() if c > 0])

    @property
    def majority_class(self) -> str | None:
        if not self.class_counts:
            return None
        return max(self.class_counts.items(), key=lambda item: item[1])[0]

    @property
    def minority_class(self) -> str | None:
        present = {k: v for k, v in self.class_counts.items() if v > 0}
        if not present:
            return None
        return min(present.items(), key=lambda item: item[1])[0]

    @property
    def minority_count(self) -> int:
        present = [v for v in self.class_counts.values() if v > 0]
        return min(present) if present else 0

    @property
    def imbalance_ratio(self) -> float:
        if self.minority_count == 0:
            return float("inf")
        return max(self.class_counts.values()) / self.minority_count

    @property
    def baseline_accuracy(self) -> float:
        """아무것도 배우지 않고 다수 클래스만 찍었을 때의 정확도."""
        if self.total_count == 0:
            return 0.0
        return max(self.class_counts.values()) / self.total_count

    @property
    def minority_share(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.minority_count / self.total_count

    @property
    def expected_minority_in_test(self) -> float:
        """평가 집합에 남을 소수 클래스 표본 수의 기댓값."""
        return self.minority_count * self.test_split_ratio


@dataclass(frozen=True, slots=True)
class BalancePolicy:
    max_imbalance_ratio: float = 20.0
    min_minority_count: int = 100
    min_expected_minority_in_test: float = 20.0
    max_baseline_accuracy: float = 0.95
    """다수 클래스만 찍어서 95% 를 넘으면 정확도 지표 자체를 쓸 수 없다."""

    min_class_count: int = 2

    def __post_init__(self) -> None:
        if self.max_imbalance_ratio <= 1.0:
            raise InvariantViolation(
                "max_imbalance_ratio 는 1보다 커야 한다.", subject="max_imbalance_ratio"
            )
        if not 0.0 < self.max_baseline_accuracy < 1.0:
            raise InvariantViolation(
                "max_baseline_accuracy 는 0과 1 사이여야 한다.",
                subject="max_baseline_accuracy",
            )

    def evaluate(self, measurement: ClassBalanceMeasurement) -> DimensionResult:
        findings: list[Finding] = []
        deductions: list[float] = []

        if measurement.class_count < self.min_class_count:
            return DimensionResult(
                dimension=QualityDimension.BALANCE,
                score=QualityScore(0.0),
                findings=(
                    Finding(
                        code="BALANCE_SINGLE_CLASS",
                        message="실제로 등장하는 클래스가 하나뿐이다. 분류 문제가 성립하지 않는다.",
                        severity=Severity.CRITICAL,
                        subject="class",
                        measured=float(measurement.class_count),
                        threshold=float(self.min_class_count),
                    ),
                ),
            )

        ratio = measurement.imbalance_ratio
        if ratio > self.max_imbalance_ratio:
            findings.append(
                Finding(
                    code="BALANCE_IMBALANCED",
                    message=(
                        f"다수 클래스가 소수 클래스의 {ratio:.0f}배다. "
                        "손실 가중치나 재표집 없이 학습하면 소수 클래스는 무시된다."
                    ),
                    severity=Severity.WARNING,
                    subject=measurement.minority_class,
                    measured=ratio,
                    threshold=self.max_imbalance_ratio,
                )
            )
            deductions.append(
                deduct(
                    min(ratio, 1000.0),
                    tolerance=self.max_imbalance_ratio,
                    cap=200.0,
                    weight=25.0,
                )
            )

        if measurement.baseline_accuracy > self.max_baseline_accuracy:
            findings.append(
                Finding(
                    code="BALANCE_BASELINE_TOO_HIGH",
                    message=(
                        f"아무것도 배우지 않고 '{measurement.majority_class}' 만 찍어도 "
                        f"정확도 {measurement.baseline_accuracy:.2%} 가 나온다. "
                        "정확도 대신 재현율·PR-AUC 로 평가해야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="accuracy",
                    measured=measurement.baseline_accuracy,
                    threshold=self.max_baseline_accuracy,
                )
            )
            deductions.append(
                deduct(
                    measurement.baseline_accuracy,
                    tolerance=self.max_baseline_accuracy,
                    cap=1.0,
                    weight=35.0,
                )
            )

        if measurement.minority_count < self.min_minority_count:
            findings.append(
                Finding(
                    code="BALANCE_MINORITY_TOO_FEW",
                    message=(
                        f"'{measurement.minority_class}' 의 절대 표본 수가 "
                        f"{measurement.minority_count} 개다. "
                        "비율이 아니라 개수가 부족하면 재표집으로도 해결되지 않는다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=measurement.minority_class,
                    measured=float(measurement.minority_count),
                    threshold=float(self.min_minority_count),
                )
            )
            deductions.append(
                deduct(
                    self.min_minority_count - measurement.minority_count,
                    tolerance=0.0,
                    cap=float(self.min_minority_count),
                    weight=30.0,
                )
            )

        expected = measurement.expected_minority_in_test
        if expected < self.min_expected_minority_in_test:
            findings.append(
                Finding(
                    code="BALANCE_TEST_TOO_THIN",
                    message=(
                        f"분할하면 평가 집합에 '{measurement.minority_class}' 가 "
                        f"{expected:.0f} 개밖에 남지 않는다. "
                        "그 숫자로 계산한 재현율은 오차가 너무 크다."
                    ),
                    severity=Severity.WARNING,
                    subject=measurement.minority_class,
                    measured=expected,
                    threshold=self.min_expected_minority_in_test,
                )
            )
            deductions.append(
                deduct(
                    self.min_expected_minority_in_test - expected,
                    tolerance=0.0,
                    cap=self.min_expected_minority_in_test,
                    weight=10.0,
                )
            )

        return DimensionResult(
            dimension=QualityDimension.BALANCE,
            score=QualityScore.from_deductions(deductions),
            findings=tuple(findings),
        )
