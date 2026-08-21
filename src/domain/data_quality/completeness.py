"""결측 — 값이 없다. (실습 2-2)

핵심: **결측률이 같아도 패턴이 다르면 대응이 다르다.**

    3% 가 고르게 흩어져 있다        → 보간해도 된다
    3% 가 한 구간에 몰려 있다        → 그 구간은 센서가 죽었다. 버려야 한다
    3% 가 특정 LOT 에만 있다         → 그 LOT 의 수집 자체가 의심스럽다

그리고 가장 위험한 것은 **숨겨진 결측**이다.
`fillna(0)` 한 줄이면 결측률은 0%가 되고, 26℃ 짜리 공장에 0℃ 가 생긴다.
그 0℃ 는 물리 범위 안에 있어서 어떤 범위 검사도 통과한다.
"""

from __future__ import annotations

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
class FieldMissingness:
    """열 하나의 결측 실태."""

    field_name: str
    total_count: int
    missing_count: int = 0
    longest_missing_run: int = 0
    """연속으로 비어 있는 최대 길이. 센서 단절의 흔적이다."""

    concentration_ratio: float = 0.0
    """결측이 특정 그룹에 몰린 정도. 0=고르게 흩어짐, 1=한 그룹에 전부."""

    repeated_value: float | None = None
    """연속형 열에서 비정상적으로 자주 등장하는 정확한 값 (은폐 결측 후보)."""

    repeated_value_count: int = 0
    repeated_value_mean_run: float = 1.0
    """그 값이 연속으로 이어진 평균 길이.

    이 숫자가 은폐 결측과 진짜 물리 상태를 가른다.
        1에 가까우면  → 무작위로 흩어져 있다. 사람이 채워 넣은 값이다.
        크면          → 뭉쳐 있다. 설비가 실제로 그 상태였다 (예: 정지 중 rpm=0).
    """

    def __post_init__(self) -> None:
        if self.total_count < 0:
            raise InvariantViolation("total_count 는 음수일 수 없다.", subject=self.field_name)
        for name, value in (
            ("missing_count", self.missing_count),
            ("longest_missing_run", self.longest_missing_run),
            ("repeated_value_count", self.repeated_value_count),
        ):
            if value < 0 or value > self.total_count:
                raise InvariantViolation(
                    f"{name}({value}) 이 total_count({self.total_count}) 범위를 벗어났다.",
                    subject=self.field_name,
                )
        if not 0.0 <= self.concentration_ratio <= 1.0:
            raise InvariantViolation(
                "concentration_ratio 는 0~1 이어야 한다.", subject=self.field_name
            )

    def _ratio(self, count: int) -> float:
        return count / self.total_count if self.total_count else 0.0

    @property
    def missing_ratio(self) -> float:
        return self._ratio(self.missing_count)

    @property
    def longest_run_ratio(self) -> float:
        return self._ratio(self.longest_missing_run)

    @property
    def repeated_value_ratio(self) -> float:
        return self._ratio(self.repeated_value_count)

    def hidden_missing_suspected(self, max_scattered_run: float = 2.0) -> bool:
        """반복되는 값이 '흩어져' 있으면 채워 넣은 값으로 의심한다."""
        return (
            self.repeated_value is not None
            and self.repeated_value_count > 0
            and self.repeated_value_mean_run < max_scattered_run
        )


@dataclass(frozen=True, slots=True)
class MissingValueMeasurement:
    fields: tuple[FieldMissingness, ...] = field(default_factory=tuple)

    @property
    def worst_missing_ratio(self) -> float:
        return max((f.missing_ratio for f in self.fields), default=0.0)

    @property
    def total_rows(self) -> int:
        return max((f.total_count for f in self.fields), default=0)

    def field_of(self, name: str) -> FieldMissingness | None:
        for item in self.fields:
            if item.field_name == name:
                return item
        return None

    @property
    def rows_with_any_missing_upper_bound(self) -> int:
        """모든 결측이 서로 다른 행에 있다고 가정한 상한."""
        return min(sum(f.missing_count for f in self.fields), self.total_rows)


@dataclass(frozen=True, slots=True)
class CompletenessPolicy:
    max_missing_ratio: float = 0.02
    max_missing_run_ratio: float = 0.01
    """전체의 1%를 넘는 연속 결측은 '구간 소실'로 본다."""

    min_missing_run_length: int = 10
    max_concentration_ratio: float = 0.6
    """결측의 60% 이상이 한 그룹에 몰리면 무작위 결측이 아니다."""

    max_repeated_value_ratio: float = 0.01
    """연속형 열에서 같은 값이 1%를 넘게 반복되면 은폐 결측을 의심한다."""

    max_scattered_run_length: float = 2.0
    """반복 값의 평균 연속 길이가 이보다 짧으면 '흩어져 있다'고 본다.

    설비 정지 중의 rpm=0 은 뭉쳐서 나타나므로 여기 걸리지 않는다.
    fillna(0) 로 채운 값은 무작위로 흩어지므로 걸린다."""

    def __post_init__(self) -> None:
        for name in (
            "max_missing_ratio",
            "max_missing_run_ratio",
            "max_concentration_ratio",
            "max_repeated_value_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 비율이어야 한다.", subject=name)

    def evaluate(self, measurement: MissingValueMeasurement) -> DimensionResult:
        """점수는 **가장 나쁜 필드**로 낸다.

        여섯 채널 중 하나가 못 쓰게 되면, 그 데이터는 그 채널을 못 쓴다.
        평균을 내면 그 사실이 5/6 만큼 희석되어 사라진다.
        """
        findings: list[Finding] = []
        per_field: dict[str, float] = {}

        def penalize(name: str, amount: float) -> None:
            per_field[name] = per_field.get(name, 0.0) + amount

        for item in measurement.fields:
            per_field.setdefault(item.field_name, 0.0)
            if item.missing_ratio > self.max_missing_ratio:
                findings.append(
                    Finding(
                        code="MISSING_RATIO_HIGH",
                        message="비어 있는 값이 기준을 넘는다.",
                        severity=Severity.CRITICAL
                        if item.missing_ratio > self.max_missing_ratio * 3
                        else Severity.WARNING,
                        subject=item.field_name,
                        measured=item.missing_ratio,
                        threshold=self.max_missing_ratio,
                    )
                )
            penalize(
                item.field_name,
                deduct(
                    item.missing_ratio,
                    tolerance=self.max_missing_ratio,
                    cap=0.30,
                    weight=45.0,
                ),
            )

            if (
                item.longest_missing_run >= self.min_missing_run_length
                and item.longest_run_ratio > self.max_missing_run_ratio
            ):
                findings.append(
                    Finding(
                        code="MISSING_RUN_LONG",
                        message=(
                            f"{item.longest_missing_run} 표본이 연속으로 비어 있다. "
                            "보간으로 채울 구간이 아니라 잘라낼 구간이다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=item.field_name,
                        measured=item.longest_run_ratio,
                        threshold=self.max_missing_run_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        item.longest_run_ratio,
                        tolerance=self.max_missing_run_ratio,
                        cap=0.10,
                        weight=25.0,
                    ),
                )

            if (
                item.missing_count > 0
                and item.concentration_ratio > self.max_concentration_ratio
            ):
                findings.append(
                    Finding(
                        code="MISSING_CONCENTRATED",
                        message=(
                            "결측이 특정 구간/그룹에 몰려 있다. 무작위 결측이 아니므로 "
                            "평균으로 채우면 그 구간의 현실이 통째로 조작된다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=item.field_name,
                        measured=item.concentration_ratio,
                        threshold=self.max_concentration_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        item.concentration_ratio,
                        tolerance=self.max_concentration_ratio,
                        cap=1.0,
                        weight=20.0,
                    ),
                )

            if item.repeated_value_ratio > self.max_repeated_value_ratio and (
                item.hidden_missing_suspected(self.max_scattered_run_length)
            ):
                findings.append(
                    Finding(
                        code="MISSING_HIDDEN",
                        message=(
                            f"연속형 열에 값 {item.repeated_value!r} 이 "
                            f"{item.repeated_value_count} 번 반복되는데, "
                            f"평균 연속 길이가 {item.repeated_value_mean_run:.1f} 로 흩어져 있다. "
                            "결측을 채워 넣은 흔적이다 — 물리 범위 검사로는 잡히지 않는다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=item.field_name,
                        measured=item.repeated_value_ratio,
                        threshold=self.max_repeated_value_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        item.repeated_value_ratio,
                        tolerance=self.max_repeated_value_ratio,
                        cap=0.15,
                        weight=40.0,
                    ),
                )

        return DimensionResult(
            dimension=QualityDimension.COMPLETENESS,
            score=QualityScore.from_deductions([max(per_field.values(), default=0.0)]),
            findings=tuple(findings),
        )
