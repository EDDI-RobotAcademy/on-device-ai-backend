"""시간축(Value Object + Policy).

실습 1-5 "시간축을 놓치면 현장을 잃는다".

시계열에서 행의 순서는 장식이 아니라 데이터의 절반이다.
    - 타임스탬프가 뒤섞이면 "원인 → 결과"가 "결과 → 원인"이 된다.
    - 같은 시각이 두 번 있으면 그 순간 설비가 두 개였다는 뜻이 된다.
    - 10초 주기 데이터에 3분 구멍이 있으면, 그 구멍이 바로 사고 구간인 경우가 많다.

pandas 는 이런 사실을 알려주지 않는다. 정렬해 버리고 조용히 넘어간다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from domain.data.inspection import Finding, InspectionKind, InspectionReport, Severity
from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class SamplingInterval:
    """현장이 약속한 수집 주기."""

    seconds: float

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise InvariantViolation("수집 주기는 0보다 커야 한다.", subject="seconds")

    @classmethod
    def from_hertz(cls, hertz: float) -> SamplingInterval:
        if hertz <= 0:
            raise InvariantViolation("샘플링 주파수는 0보다 커야 한다.", subject="hertz")
        return cls(seconds=1.0 / hertz)

    @property
    def hertz(self) -> float:
        return 1.0 / self.seconds

    def expected_count(self, duration_seconds: float) -> int:
        """이 구간이라면 원래 몇 개가 있어야 하는가."""
        if duration_seconds < 0:
            raise InvariantViolation("기간은 음수일 수 없다.", subject="duration_seconds")
        return int(duration_seconds // self.seconds) + 1


@dataclass(frozen=True, slots=True)
class TimeAxisMeasurement:
    """시간축에 대해 Infrastructure 가 측정한 값."""

    field_name: str
    record_count: int
    first: datetime
    last: datetime
    median_interval_seconds: float
    out_of_order_count: int = 0
    duplicate_timestamp_count: int = 0
    gap_count: int = 0
    longest_gap_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.record_count < 0:
            raise InvariantViolation("record_count 는 음수일 수 없다.", subject=self.field_name)
        if self.last < self.first:
            raise InvariantViolation(
                "마지막 시각이 첫 시각보다 앞설 수 없다.", subject=self.field_name
            )
        if self.median_interval_seconds < 0:
            raise InvariantViolation(
                "median_interval_seconds 는 음수일 수 없다.", subject=self.field_name
            )

    @property
    def duration_seconds(self) -> float:
        return (self.last - self.first).total_seconds()

    @property
    def out_of_order_ratio(self) -> float:
        return self.out_of_order_count / self.record_count if self.record_count else 0.0

    @property
    def duplicate_ratio(self) -> float:
        return self.duplicate_timestamp_count / self.record_count if self.record_count else 0.0

    def coverage_ratio(self, interval: SamplingInterval) -> float:
        """약속한 주기 기준으로 실제로 몇 %가 남아 있는가."""
        expected = interval.expected_count(self.duration_seconds)
        if expected == 0:
            return 0.0
        return min(self.record_count / expected, 1.0)


@dataclass(frozen=True, slots=True)
class TimeAxisPolicy:
    """시간축이 신뢰할 만한지에 대한 기준."""

    expected_interval: SamplingInterval
    interval_tolerance_ratio: float = 0.2
    """실측 중앙 간격이 약속 주기에서 ±20% 를 벗어나면 수집 설정이 다르다."""

    max_gap_multiplier: float = 3.0
    """약속 주기의 3배를 넘으면 결측 구간으로 본다."""

    max_duplicate_ratio: float = 0.0
    allow_out_of_order: bool = False
    min_coverage_ratio: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.interval_tolerance_ratio < 1.0:
            raise InvariantViolation(
                "interval_tolerance_ratio 는 0 이상 1 미만이어야 한다.",
                subject="interval_tolerance_ratio",
            )
        if self.max_gap_multiplier <= 1.0:
            raise InvariantViolation(
                "max_gap_multiplier 는 1보다 커야 한다.", subject="max_gap_multiplier"
            )
        if not 0.0 <= self.min_coverage_ratio <= 1.0:
            raise InvariantViolation(
                "min_coverage_ratio 는 0~1 이어야 한다.", subject="min_coverage_ratio"
            )

    def inspect(self, measurement: TimeAxisMeasurement) -> InspectionReport:
        m = measurement
        findings: list[Finding] = []

        if not self.allow_out_of_order and m.out_of_order_count > 0:
            findings.append(
                Finding(
                    code="TIME_OUT_OF_ORDER",
                    message=(
                        f"{m.out_of_order_count} 개 행이 시간 역순이다. "
                        "정렬로 덮으면 인과가 뒤집힌 채 학습된다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=m.field_name,
                    measured=m.out_of_order_ratio,
                    threshold=0.0,
                )
            )

        if m.duplicate_ratio > self.max_duplicate_ratio:
            findings.append(
                Finding(
                    code="TIME_DUPLICATED",
                    message=(
                        f"같은 시각이 {m.duplicate_timestamp_count} 번 중복된다. "
                        "한 설비가 같은 순간에 두 값을 가질 수는 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=m.field_name,
                    measured=m.duplicate_ratio,
                    threshold=self.max_duplicate_ratio,
                )
            )

        expected = self.expected_interval.seconds
        if expected > 0 and m.median_interval_seconds > 0:
            deviation = abs(m.median_interval_seconds - expected) / expected
            if deviation > self.interval_tolerance_ratio:
                findings.append(
                    Finding(
                        code="TIME_INTERVAL_MISMATCH",
                        message=(
                            f"약속 주기 {expected:g}s 인데 실측 중앙 간격은 "
                            f"{m.median_interval_seconds:g}s 다."
                        ),
                        severity=Severity.WARNING,
                        subject=m.field_name,
                        measured=m.median_interval_seconds,
                        threshold=expected,
                    )
                )

        gap_threshold = expected * self.max_gap_multiplier
        if m.longest_gap_seconds > gap_threshold:
            findings.append(
                Finding(
                    code="TIME_GAP",
                    message=(
                        f"{m.gap_count} 개 구간이 끊겼다. 가장 긴 공백은 "
                        f"{m.longest_gap_seconds:g}s 다. 그 시간에 현장에서 무슨 일이 있었는지 "
                        "데이터에는 남아 있지 않다."
                    ),
                    severity=Severity.WARNING,
                    subject=m.field_name,
                    measured=m.longest_gap_seconds,
                    threshold=gap_threshold,
                )
            )

        coverage = m.coverage_ratio(self.expected_interval)
        if coverage < self.min_coverage_ratio:
            findings.append(
                Finding(
                    code="TIME_COVERAGE_LOW",
                    message="약속 주기 기준으로 있어야 할 표본이 상당수 없다.",
                    severity=Severity.CRITICAL,
                    subject=m.field_name,
                    measured=coverage,
                    threshold=self.min_coverage_ratio,
                )
            )

        return InspectionReport(kind=InspectionKind.TIME_AXIS, findings=tuple(findings))
