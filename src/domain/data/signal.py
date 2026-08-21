"""센서와 이미지가 거짓말하는 방식(Value Object + Policy).

실습 1-4 "센서와 이미지는 거짓말을 어떻게 하는가?".

센서는 값을 비우면서 거짓말하지 않는다. **그럴듯한 값을 계속 뱉으면서** 거짓말한다.
    - 고착(stuck): 케이블이 빠져도 마지막 값을 계속 유지한다.
    - 포화(saturation): 측정 한계에 붙어 실제 값을 잘라먹는다.
    - 범위 이탈: 물리적으로 불가능한 값을 정상처럼 기록한다.

이미지는 다르게 거짓말한다.
    - 조명이 바뀌어 밝기 분포가 통째로 이동한다.
    - 초점이 나가 결함 자체가 화면에서 사라진다.
    - 같은 사진이 다른 이름으로 여러 장 들어온다.

결측(NaN)은 눈에 보이므로 오히려 덜 위험하다. 여기서 잡는 것은 *보이지 않는 거짓말*이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.data.inspection import Finding, InspectionKind, InspectionReport, Severity
from domain.shared.errors import InvariantViolation


class SignalDefect(Enum):
    STUCK = "STUCK"
    SATURATED = "SATURATED"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    UNREADABLE = "UNREADABLE"
    DEFOCUSED = "DEFOCUSED"
    EXPOSURE_SHIFT = "EXPOSURE_SHIFT"
    VISUAL_DUPLICATE = "VISUAL_DUPLICATE"


@dataclass(frozen=True, slots=True)
class SensorChannelMeasurement:
    """센서 채널 하나에 대해 Infrastructure 가 측정한 값."""

    field_name: str
    total_count: int
    out_of_range_count: int = 0
    """FieldSpec.value_range 를 벗어난 표본 수."""

    longest_constant_run: int = 0
    """값이 한 번도 변하지 않고 연속된 최대 길이."""

    saturated_count: int = 0
    """관측 최소/최대값에 정확히 붙어 있는 표본 수."""

    def __post_init__(self) -> None:
        if self.total_count < 0:
            raise InvariantViolation("total_count 는 음수일 수 없다.", subject=self.field_name)
        for name, value in (
            ("out_of_range_count", self.out_of_range_count),
            ("saturated_count", self.saturated_count),
            ("longest_constant_run", self.longest_constant_run),
        ):
            if value < 0 or value > self.total_count:
                raise InvariantViolation(
                    f"{name}({value}) 이 total_count({self.total_count}) 범위를 벗어났다.",
                    subject=self.field_name,
                )

    def _ratio(self, count: int) -> float:
        return count / self.total_count if self.total_count else 0.0

    @property
    def out_of_range_ratio(self) -> float:
        return self._ratio(self.out_of_range_count)

    @property
    def saturated_ratio(self) -> float:
        return self._ratio(self.saturated_count)

    @property
    def constant_run_ratio(self) -> float:
        return self._ratio(self.longest_constant_run)


@dataclass(frozen=True, slots=True)
class ImageIntegrityMeasurement:
    """이미지 데이터셋에 대해 Infrastructure 가 측정한 값.

    "몇 장이 흐린가"를 측정기가 세지 않는다. 점수만 넘긴다.
    흐림의 기준선은 현장마다 다르고, 그것은 Policy 의 몫이다.
    """

    total_images: int
    unreadable_count: int = 0
    focus_scores: tuple[float, ...] = field(default_factory=tuple)
    """이미지별 Laplacian variance. 작을수록 흐리다."""

    brightness_values: tuple[float, ...] = field(default_factory=tuple)
    """이미지별 평균 밝기 (0.0 ~ 255.0)."""

    visual_duplicate_count: int = 0
    """시각적으로 같은 이미지가 몇 장 겹쳤는가 (원본 1장은 제외)."""

    distinct_resolution_count: int = 1

    def __post_init__(self) -> None:
        if self.total_images < 0:
            raise InvariantViolation("total_images 는 음수일 수 없다.", subject="total_images")
        if self.distinct_resolution_count < 0:
            raise InvariantViolation(
                "distinct_resolution_count 는 음수일 수 없다.",
                subject="distinct_resolution_count",
            )
        for name, value in (
            ("unreadable_count", self.unreadable_count),
            ("visual_duplicate_count", self.visual_duplicate_count),
        ):
            if value < 0 or value > self.total_images:
                raise InvariantViolation(
                    f"{name}({value}) 이 total_images({self.total_images}) 범위를 벗어났다.",
                    subject=name,
                )

    def _ratio(self, count: int) -> float:
        return count / self.total_images if self.total_images else 0.0

    @property
    def readable_count(self) -> int:
        return self.total_images - self.unreadable_count

    @property
    def unreadable_ratio(self) -> float:
        return self._ratio(self.unreadable_count)

    @property
    def visual_duplicate_ratio(self) -> float:
        return self._ratio(self.visual_duplicate_count)

    @property
    def min_focus_score(self) -> float:
        return min(self.focus_scores) if self.focus_scores else 0.0

    def defocused_count(self, threshold: float) -> int:
        return sum(1 for s in self.focus_scores if s < threshold)

    def defocused_ratio(self, threshold: float) -> float:
        return self._ratio(self.defocused_count(threshold))

    @property
    def mean_brightness(self) -> float:
        if not self.brightness_values:
            return 0.0
        return sum(self.brightness_values) / len(self.brightness_values)

    @property
    def brightness_stddev(self) -> float:
        values = self.brightness_values
        if len(values) < 2:
            return 0.0
        mean = self.mean_brightness
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance**0.5


@dataclass(frozen=True, slots=True)
class SignalPlausibilityPolicy:
    """신호가 물리적으로 말이 되는지에 대한 기준.

    이 숫자들은 코드가 정하는 것이 아니라 현장이 정한다.
    그래서 Policy 는 Value Object 로 주입되고, 라인마다 다를 수 있다.
    """

    max_out_of_range_ratio: float = 0.0
    """물리 범위 이탈은 원칙적으로 0 이어야 한다."""

    max_constant_run_ratio: float = 0.05
    """전체의 5% 를 넘게 값이 고정되면 센서 고착을 의심한다."""

    min_constant_run_length: int = 10
    """이보다 짧은 연속은 정상 운전으로 본다."""

    max_saturated_ratio: float = 0.02
    min_focus_score: float = 50.0
    max_unreadable_ratio: float = 0.0
    max_defocused_ratio: float = 0.05
    max_visual_duplicate_ratio: float = 0.02
    max_brightness_stddev: float = 40.0

    def __post_init__(self) -> None:
        for name in (
            "max_out_of_range_ratio",
            "max_constant_run_ratio",
            "max_saturated_ratio",
            "max_unreadable_ratio",
            "max_defocused_ratio",
            "max_visual_duplicate_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 비율이어야 한다.", subject=name)

    # -- 센서 --------------------------------------------------------------
    def inspect_sensors(
        self, measurements: tuple[SensorChannelMeasurement, ...]
    ) -> InspectionReport:
        findings: list[Finding] = []
        for m in measurements:
            if m.out_of_range_ratio > self.max_out_of_range_ratio:
                findings.append(
                    Finding(
                        code=f"SIGNAL_{SignalDefect.OUT_OF_RANGE.value}",
                        message="물리적으로 불가능한 값이 정상 표본처럼 기록되어 있다.",
                        severity=Severity.CRITICAL,
                        subject=m.field_name,
                        measured=m.out_of_range_ratio,
                        threshold=self.max_out_of_range_ratio,
                    )
                )
            if (
                m.longest_constant_run >= self.min_constant_run_length
                and m.constant_run_ratio > self.max_constant_run_ratio
            ):
                findings.append(
                    Finding(
                        code=f"SIGNAL_{SignalDefect.STUCK.value}",
                        message=(
                            f"값이 {m.longest_constant_run} 표본 동안 변하지 않았다. "
                            "센서 고착 또는 통신 두절을 의심한다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=m.field_name,
                        measured=m.constant_run_ratio,
                        threshold=self.max_constant_run_ratio,
                    )
                )
            if m.saturated_ratio > self.max_saturated_ratio:
                findings.append(
                    Finding(
                        code=f"SIGNAL_{SignalDefect.SATURATED.value}",
                        message="측정 한계에 붙어 실제 값이 잘려 있다. 큰 값일수록 정보가 사라진다.",
                        severity=Severity.WARNING,
                        subject=m.field_name,
                        measured=m.saturated_ratio,
                        threshold=self.max_saturated_ratio,
                    )
                )
        return InspectionReport(
            kind=InspectionKind.SIGNAL_PLAUSIBILITY, findings=tuple(findings)
        )

    # -- 이미지 ------------------------------------------------------------
    def inspect_images(self, measurement: ImageIntegrityMeasurement) -> InspectionReport:
        m = measurement
        findings: list[Finding] = []
        if m.unreadable_ratio > self.max_unreadable_ratio:
            findings.append(
                Finding(
                    code=f"SIGNAL_{SignalDefect.UNREADABLE.value}",
                    message="열리지 않는 이미지가 있다. 수집 경로 어딘가에서 파일이 깨졌다.",
                    severity=Severity.CRITICAL,
                    subject="image",
                    measured=m.unreadable_ratio,
                    threshold=self.max_unreadable_ratio,
                )
            )
        defocused_ratio = m.defocused_ratio(self.min_focus_score)
        if defocused_ratio > self.max_defocused_ratio:
            findings.append(
                Finding(
                    code=f"SIGNAL_{SignalDefect.DEFOCUSED.value}",
                    message=(
                        f"초점이 나간 이미지가 {m.defocused_count(self.min_focus_score)} 장이다. "
                        "결함이 화면에서 사라진 상태로 학습된다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="focus",
                    measured=defocused_ratio,
                    threshold=self.max_defocused_ratio,
                )
            )
        elif m.min_focus_score < self.min_focus_score:
            findings.append(
                Finding(
                    code=f"SIGNAL_{SignalDefect.DEFOCUSED.value}",
                    message="가장 흐린 이미지가 기준 이하다.",
                    severity=Severity.WARNING,
                    subject="focus",
                    measured=m.min_focus_score,
                    threshold=self.min_focus_score,
                )
            )
        if m.brightness_stddev > self.max_brightness_stddev:
            findings.append(
                Finding(
                    code=f"SIGNAL_{SignalDefect.EXPOSURE_SHIFT.value}",
                    message="밝기 산포가 크다. 촬영 조명이 도중에 바뀌었을 가능성이 높다.",
                    severity=Severity.WARNING,
                    subject="brightness",
                    measured=m.brightness_stddev,
                    threshold=self.max_brightness_stddev,
                )
            )
        if m.visual_duplicate_ratio > self.max_visual_duplicate_ratio:
            findings.append(
                Finding(
                    code=f"SIGNAL_{SignalDefect.VISUAL_DUPLICATE.value}",
                    message="사실상 같은 이미지가 여러 장이다. 분할 시 학습/평가에 나뉘어 들어가면 성능이 부풀려진다.",
                    severity=Severity.WARNING,
                    subject="duplicate",
                    measured=m.visual_duplicate_ratio,
                    threshold=self.max_visual_duplicate_ratio,
                )
            )
        if m.distinct_resolution_count > 1:
            findings.append(
                Finding(
                    code="SIGNAL_RESOLUTION_MIXED",
                    message=f"해상도가 {m.distinct_resolution_count} 종류로 섞여 있다.",
                    severity=Severity.WARNING,
                    subject="resolution",
                    measured=float(m.distinct_resolution_count),
                    threshold=1.0,
                )
            )
        return InspectionReport(
            kind=InspectionKind.SIGNAL_PLAUSIBILITY, findings=tuple(findings)
        )
