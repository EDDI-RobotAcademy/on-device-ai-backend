"""AI가 먹을 수 있는 형태의 설계(Value Object).

실습 1-7 "AI가 먹을 수 있는 데이터로 다시 설계하라".

CSV 한 줄은 모델의 입력이 아니다. 모델의 입력은 고정된 모양(shape)을 가진 텐서다.
시계열이라면 "몇 초를 한 덩어리로 볼 것인가"를 정해야 그 모양이 결정된다.

여기서 정하는 것은 값이 아니라 **계약**이다.
    - 어떤 필드가 입력인가 (ID 를 넣으면 모델은 ID 를 외운다)
    - 몇 개를 한 창(window)으로 묶는가
    - 정규화 통계는 어디서 뽑는가  ← train 이 아니면 그 순간 누수다

이 계약은 이후 학습·최적화·배포에서 전부 다시 쓰인다.
전처리 코드가 학습 스크립트와 디바이스에서 어긋나는 사고는 여기서 막는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.data.errors import UnknownField
from domain.data.inspection import Finding, InspectionKind, InspectionReport, Severity
from domain.data.schema import DataSchema, FieldRole
from domain.data.time_axis import SamplingInterval
from domain.shared.errors import InvariantViolation


class NormalizationMethod(Enum):
    NONE = "NONE"
    ZSCORE = "ZSCORE"
    MINMAX = "MINMAX"


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """시계열을 자르는 창."""

    length: int
    stride: int
    interval: SamplingInterval

    def __post_init__(self) -> None:
        if self.length < 1:
            raise InvariantViolation("window length 는 1 이상이어야 한다.", subject="length")
        if self.stride < 1:
            raise InvariantViolation("stride 는 1 이상이어야 한다.", subject="stride")
        if self.stride > self.length:
            raise InvariantViolation(
                f"stride({self.stride}) 가 length({self.length}) 보다 크면 표본 사이가 통째로 버려진다.",
                subject="stride",
            )

    @property
    def duration_seconds(self) -> float:
        """이 창이 현장 시간으로 몇 초를 보는가."""
        return self.length * self.interval.seconds

    @property
    def overlap_ratio(self) -> float:
        return 1.0 - self.stride / self.length


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    """정규화 계약.

    statistics 는 반드시 train 분할에서만 계산한다.
    전체 데이터로 평균을 내면, 모델은 시험 문제를 미리 본 셈이 된다.
    """

    method: NormalizationMethod
    fitted_on: str = "train"
    statistics: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    """필드명 → (중심, 스케일). ZSCORE 면 (mean, std), MINMAX 면 (min, max)."""

    def __post_init__(self) -> None:
        if self.method is NormalizationMethod.NONE:
            return
        if self.fitted_on != "train":
            raise InvariantViolation(
                f"정규화 통계를 '{self.fitted_on}' 에서 뽑았다. train 이 아니면 데이터 누수다.",
                subject="fitted_on",
            )
        for name, (center, scale) in self.statistics.items():
            if self.method is NormalizationMethod.ZSCORE and scale <= 0:
                raise InvariantViolation(
                    f"'{name}' 의 표준편차가 0 이다. 값이 변하지 않는 열을 정규화할 수 없다.",
                    subject=name,
                )
            if self.method is NormalizationMethod.MINMAX and scale <= center:
                raise InvariantViolation(
                    f"'{name}' 의 max 가 min 보다 크지 않다.", subject=name
                )


@dataclass(frozen=True, slots=True)
class ImageInputSpec:
    """이미지 입력 계약."""

    width: int
    height: int
    channels: int = 3

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise InvariantViolation("이미지 크기는 1 이상이어야 한다.", subject="size")
        if self.channels not in (1, 3):
            raise InvariantViolation(
                "채널은 1(GRAY) 또는 3(RGB) 만 지원한다.", subject="channels"
            )


@dataclass(frozen=True, slots=True)
class TrainingDataSpec:
    """"이 데이터를 이렇게 잘라서 모델에 넣는다"는 계약."""

    schema: DataSchema
    feature_fields: tuple[str, ...]
    label_field: str
    window: WindowSpec | None = None
    image: ImageInputSpec | None = None
    normalization: NormalizationSpec = field(
        default_factory=lambda: NormalizationSpec(method=NormalizationMethod.NONE)
    )

    def __post_init__(self) -> None:
        if not self.feature_fields:
            raise InvariantViolation("입력 필드가 없다.", subject="feature_fields")
        if len(self.feature_fields) != len(set(self.feature_fields)):
            raise InvariantViolation("입력 필드가 중복되었다.", subject="feature_fields")

        for name in self.feature_fields:
            spec = self.schema.field_of(name)  # 없으면 UnknownField
            if spec.role is FieldRole.IDENTIFIER:
                raise InvariantViolation(
                    f"'{name}' 은 식별자다. 입력에 넣으면 모델은 패턴이 아니라 ID 를 외운다.",
                    subject=name,
                )
            if spec.role is FieldRole.LABEL:
                raise InvariantViolation(
                    f"'{name}' 은 정답이다. 입력에 넣으면 정확도 100% 짜리 쓸모없는 모델이 된다.",
                    subject=name,
                )
            if spec.role is FieldRole.GROUP:
                raise InvariantViolation(
                    f"'{name}' 은 LOT/설비 묶음 식별자다. "
                    "입력에 넣으면 모델은 공정이 아니라 그 묶음을 외운다.",
                    subject=name,
                )
            if spec.role is FieldRole.TIME_INDEX:
                raise InvariantViolation(
                    f"'{name}' 은 시간축이다. 시각 자체를 입력에 넣으면 배포 즉시 분포가 벗어난다.",
                    subject=name,
                )

        label_spec = self.schema.field_of(self.label_field)
        if label_spec.role is not FieldRole.LABEL:
            raise InvariantViolation(
                f"'{self.label_field}' 의 역할은 {label_spec.role.value} 다. 정답이 아니다.",
                subject=self.label_field,
            )

        if self.schema.time_index is not None and self.window is None and self.image is None:
            raise InvariantViolation(
                "시간축이 있는 데이터인데 창(window) 설계가 없다. 한 줄만 보는 모델은 현장을 못 본다.",
                subject="window",
            )
        if self.window is not None and self.image is not None:
            raise InvariantViolation(
                "시계열 창과 이미지 입력을 동시에 선언할 수 없다.", subject="input"
            )

        if self.normalization.method is not NormalizationMethod.NONE:
            unknown = set(self.normalization.statistics) - set(self.feature_fields)
            if unknown:
                raise UnknownField(
                    f"입력이 아닌 필드의 정규화 통계가 있다: {sorted(unknown)}",
                    subject="normalization",
                )

    @property
    def input_shape(self) -> tuple[int, ...]:
        """모델이 받게 될 텐서 모양 (배치 제외).

        시계열: (window_length, feature_count)
        이미지: (channels, height, width)   ← PyTorch 관례
        """
        if self.image is not None:
            return (self.image.channels, self.image.height, self.image.width)
        if self.window is not None:
            return (self.window.length, len(self.feature_fields))
        return (len(self.feature_fields),)

    @property
    def input_element_count(self) -> int:
        count = 1
        for dim in self.input_shape:
            count *= dim
        return count

    def inspect(self) -> InspectionReport:
        """설계가 현실적인지에 대한 경고."""
        findings: list[Finding] = []

        unused = [
            f.name
            for f in self.schema.feature_fields
            if f.name not in self.feature_fields
        ]
        for name in unused:
            findings.append(
                Finding(
                    code="SPEC_FEATURE_UNUSED",
                    message="FEATURE 로 선언했지만 입력에서 빠졌다. 의도한 것인지 확인이 필요하다.",
                    severity=Severity.INFO,
                    subject=name,
                )
            )

        if self.normalization.method is NormalizationMethod.NONE:
            findings.append(
                Finding(
                    code="SPEC_NO_NORMALIZATION",
                    message=(
                        "정규화가 없다. 단위가 다른 센서(V, A, ℃)를 그대로 넣으면 "
                        "값이 큰 축이 학습을 지배한다."
                    ),
                    severity=Severity.WARNING,
                    subject="normalization",
                )
            )
        else:
            missing_stats = [
                name for name in self.feature_fields
                if name not in self.normalization.statistics
            ]
            for name in missing_stats:
                findings.append(
                    Finding(
                        code="SPEC_STATISTICS_MISSING",
                        message="정규화를 선언했으나 이 필드의 통계가 없다. 디바이스에서 전처리가 어긋난다.",
                        severity=Severity.CRITICAL,
                        subject=name,
                    )
                )

        if self.window is not None and self.window.overlap_ratio > 0.9:
            findings.append(
                Finding(
                    code="SPEC_WINDOW_OVERLAP_HIGH",
                    message=(
                        "창이 90% 넘게 겹친다. 거의 같은 표본이 대량 복제되어 "
                        "검증 점수가 부풀려진다."
                    ),
                    severity=Severity.WARNING,
                    subject="window",
                    measured=self.window.overlap_ratio,
                    threshold=0.9,
                )
            )

        return InspectionReport(kind=InspectionKind.TRAINING_SPEC, findings=tuple(findings))
