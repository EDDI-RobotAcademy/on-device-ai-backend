"""품질의 축과 점수. (실습 2-1, 2-8)

품질은 하나의 숫자가 아니다. 서로 다른 방식으로 망가진다.
    - 값이 없다 (결측)
    - 값이 말이 안 된다 (이상치)
    - 정답이 틀렸다 (라벨 오류)
    - 한쪽으로 쏠렸다 (불균형)
    - 신호보다 잡음이 크다 (잡음)
    - 같은 것을 반복해서 가르친다 (중복)

축을 나누지 않으면 "품질이 나쁘다"까지만 말할 수 있고,
**무엇을 어떻게 고쳐야 하는지**는 말할 수 없다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import (
    Finding,
    Verdict,
    blocking_findings,
    derive_verdict,
    warning_findings,
)


class QualityDimension(Enum):
    COMPLETENESS = "COMPLETENESS"
    """값이 있는가. (실습 2-2)"""

    VALIDITY = "VALIDITY"
    """값이 말이 되는가. (실습 2-3)"""

    LABEL_QUALITY = "LABEL_QUALITY"
    """정답이 맞는가. (실습 2-4)"""

    BALANCE = "BALANCE"
    """클래스가 고른가. (실습 2-5)"""

    NOISE = "NOISE"
    """신호가 잡음에 묻히지 않았는가. (실습 2-6)"""

    UNIQUENESS = "UNIQUENESS"
    """같은 것을 반복하고 있지 않은가. (실습 2-7)"""


ALL_DIMENSIONS: tuple[QualityDimension, ...] = tuple(QualityDimension)


class QualityGrade(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


@dataclass(frozen=True, slots=True)
class QualityScore:
    """0~100 점.

    점수 자체가 목적이 아니다. 점수는 **대화를 시작하기 위한 공통 단위**다.
    "느낌상 좀 지저분한데요" 와 "COMPLETENESS 42점" 은 다른 회의를 만든다.
    """

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 100.0:
            raise InvariantViolation(
                f"품질 점수는 0~100 이어야 한다. (받은 값 {self.value})", subject="score"
            )

    @classmethod
    def perfect(cls) -> QualityScore:
        return cls(100.0)

    @classmethod
    def from_deductions(cls, deductions: Iterable[float]) -> QualityScore:
        """감점을 모아 점수를 만든다. 100점에서 시작해 깎는다."""
        total = sum(max(d, 0.0) for d in deductions)
        return cls(max(100.0 - total, 0.0))

    @property
    def grade(self) -> QualityGrade:
        if self.value >= 90:
            return QualityGrade.A
        if self.value >= 80:
            return QualityGrade.B
        if self.value >= 70:
            return QualityGrade.C
        if self.value >= 60:
            return QualityGrade.D
        return QualityGrade.F

    def __str__(self) -> str:
        return f"{self.value:.1f} ({self.grade.value})"


def deduct(
    measured: float, *, tolerance: float, cap: float, weight: float
) -> float:
    """기준 초과분에 비례하는 감점.

        measured <= tolerance   → 0점 감점
        measured >= cap         → weight 만큼 전부 감점
        그 사이                  → 선형 비례

    공식이 단순한 것은 의도다.
    점수가 왜 그렇게 나왔는지 설명할 수 없으면 아무도 그 점수를 고치려 하지 않는다.
    """
    if weight < 0:
        raise InvariantViolation("감점 가중치는 음수일 수 없다.", subject="weight")
    if cap <= tolerance:
        return weight if measured > tolerance else 0.0
    if measured <= tolerance:
        return 0.0
    ratio = (measured - tolerance) / (cap - tolerance)
    return weight * min(ratio, 1.0)


@dataclass(frozen=True, slots=True)
class DimensionResult:
    """한 축에 대한 평가 결과. 점수와 근거가 함께 다닌다."""

    dimension: QualityDimension
    score: QualityScore
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.findings)

    @property
    def passed(self) -> bool:
        return self.verdict is not Verdict.FAILED

    @property
    def blocking(self) -> tuple[Finding, ...]:
        return blocking_findings(self.findings)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return warning_findings(self.findings)

    def render(self) -> str:
        lines = [
            f"[{self.dimension.value}] {self.score}  {self.verdict.value}",
        ]
        lines += [f"  - {f.describe()}" for f in self.findings]
        if not self.findings:
            lines.append("  - (지적 사항 없음)")
        return "\n".join(lines)
