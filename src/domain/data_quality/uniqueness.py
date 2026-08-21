"""중복 — 같은 것을 반복해서 가르친다. (실습 2-7)

**행 전체가 같아야 중복이 아니다. 모델이 보는 것이 같으면 중복이다.**

타임스탬프는 모델의 입력이 아니다(실습 1-7).
그러므로 타임스탬프만 다르고 입력 값이 같은 두 행은, 모델에게는 같은 표본이다.
모듈 1의 시간축 검사는 이것을 절대 잡을 수 없다.

중복이 만드는 피해는 셋이다.
    1. 가중치 왜곡 — 중복된 구간이 여러 번 학습되어 모델이 그쪽으로 끌려간다
    2. 분할 누수 — 원본은 train, 사본은 test 로 가면 시험 문제를 미리 본 것이다 (실습 1-8)
    3. 표본 수 착시 — 8,640행인데 실제로 서로 다른 표본은 6,000개일 수 있다
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.data_quality.dimensions import (
    DimensionResult,
    QualityDimension,
    QualityScore,
    deduct,
)
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class DuplicateMeasurement:
    """입력(feature) 기준 중복 실태."""

    total_rows: int
    exact_duplicate_count: int = 0
    """입력 값이 완전히 같은 행 수 (각 묶음의 원본 1개는 제외)."""

    duplicate_group_count: int = 0
    near_duplicate_count: int = 0
    """인접 행과 사실상 같은 값인 행 수 (센서 홀드/재전송)."""

    conflicting_label_count: int = 0
    """입력은 같은데 라벨이 다른 행 수."""

    def __post_init__(self) -> None:
        if self.total_rows < 0:
            raise InvariantViolation("total_rows 는 음수일 수 없다.", subject="total_rows")
        for name in (
            "exact_duplicate_count",
            "near_duplicate_count",
            "conflicting_label_count",
        ):
            value = getattr(self, name)
            if value < 0 or value > self.total_rows:
                raise InvariantViolation(
                    f"{name}({value}) 이 total_rows({self.total_rows}) 범위를 벗어났다.",
                    subject=name,
                )

    def _ratio(self, count: int) -> float:
        return count / self.total_rows if self.total_rows else 0.0

    @property
    def exact_duplicate_ratio(self) -> float:
        return self._ratio(self.exact_duplicate_count)

    @property
    def near_duplicate_ratio(self) -> float:
        return self._ratio(self.near_duplicate_count)

    @property
    def conflicting_label_ratio(self) -> float:
        return self._ratio(self.conflicting_label_count)

    @property
    def distinct_row_count(self) -> int:
        """실제로 서로 다른 표본 수."""
        return self.total_rows - self.exact_duplicate_count

    @property
    def inflation_ratio(self) -> float:
        """표본 수가 몇 배로 부풀려져 보이는가."""
        distinct = self.distinct_row_count
        if distinct <= 0:
            return float("inf")
        return self.total_rows / distinct


@dataclass(frozen=True, slots=True)
class UniquenessPolicy:
    max_exact_duplicate_ratio: float = 0.005
    max_near_duplicate_ratio: float = 0.05
    max_conflicting_label_ratio: float = 0.0
    max_inflation_ratio: float = 1.05

    def __post_init__(self) -> None:
        for name in (
            "max_exact_duplicate_ratio",
            "max_near_duplicate_ratio",
            "max_conflicting_label_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 비율이어야 한다.", subject=name)
        if self.max_inflation_ratio < 1.0:
            raise InvariantViolation(
                "max_inflation_ratio 는 1 이상이어야 한다.", subject="max_inflation_ratio"
            )

    def evaluate(self, measurement: DuplicateMeasurement) -> DimensionResult:
        m = measurement
        findings: list[Finding] = []
        deductions: list[float] = []

        if m.exact_duplicate_ratio > self.max_exact_duplicate_ratio:
            findings.append(
                Finding(
                    code="UNIQUENESS_EXACT_DUPLICATE",
                    message=(
                        f"입력이 완전히 같은 행이 {m.exact_duplicate_count} 개 "
                        f"({m.duplicate_group_count} 묶음)다. "
                        "무작위로 나누면 원본은 train, 사본은 test 로 간다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="feature_vector",
                    measured=m.exact_duplicate_ratio,
                    threshold=self.max_exact_duplicate_ratio,
                )
            )
            deductions.append(
                deduct(
                    m.exact_duplicate_ratio,
                    tolerance=self.max_exact_duplicate_ratio,
                    cap=0.20,
                    weight=45.0,
                )
            )

        if m.near_duplicate_ratio > self.max_near_duplicate_ratio:
            findings.append(
                Finding(
                    code="UNIQUENESS_NEAR_DUPLICATE",
                    message=(
                        "인접 표본과 사실상 같은 값이 반복된다. "
                        "센서 홀드이거나 수집 재전송이다."
                    ),
                    severity=Severity.WARNING,
                    subject="feature_vector",
                    measured=m.near_duplicate_ratio,
                    threshold=self.max_near_duplicate_ratio,
                )
            )
            deductions.append(
                deduct(
                    m.near_duplicate_ratio,
                    tolerance=self.max_near_duplicate_ratio,
                    cap=0.40,
                    weight=25.0,
                )
            )

        if m.conflicting_label_ratio > self.max_conflicting_label_ratio:
            findings.append(
                Finding(
                    code="UNIQUENESS_LABEL_CONFLICT",
                    message=(
                        f"같은 입력에 다른 라벨이 붙은 행이 {m.conflicting_label_count} 개다. "
                        "중복이 라벨 오류를 드러냈다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="label",
                    measured=m.conflicting_label_ratio,
                    threshold=self.max_conflicting_label_ratio,
                )
            )
            deductions.append(
                deduct(
                    m.conflicting_label_ratio,
                    tolerance=self.max_conflicting_label_ratio,
                    cap=0.05,
                    weight=30.0,
                )
            )

        if m.inflation_ratio > self.max_inflation_ratio:
            findings.append(
                Finding(
                    code="UNIQUENESS_SAMPLE_INFLATION",
                    message=(
                        f"{m.total_rows:,}행처럼 보이지만 서로 다른 표본은 "
                        f"{m.distinct_row_count:,}개다."
                    ),
                    severity=Severity.WARNING,
                    subject="sample_count",
                    measured=m.inflation_ratio,
                    threshold=self.max_inflation_ratio,
                )
            )

        return DimensionResult(
            dimension=QualityDimension.UNIQUENESS,
            score=QualityScore.from_deductions(deductions),
            findings=tuple(findings),
        )
