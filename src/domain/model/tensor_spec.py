"""AI가 먹는 모양. (실습 3-1, 3-3)

모델은 CSV 를 먹지 않는다. **고정된 모양을 가진 숫자 덩어리**를 먹는다.

이 파일에는 Tensor 가 하나도 없다. 모양과 개수뿐이다.
실제 배열은 Infrastructure(PyTorch)가 들고 있고, Domain 은 그 **명세**만 안다.
그래서 PyTorch 를 TensorFlow 로 바꿔도 여기는 그대로다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation


class TensorLayout(Enum):
    """축의 순서. 프레임워크마다 관례가 다르다."""

    TIME_FIRST = "TIME_FIRST"
    """(시간, 채널) — 시계열의 자연스러운 순서."""

    CHANNEL_FIRST = "CHANNEL_FIRST"
    """(채널, 시간) 또는 (채널, 높이, 너비) — PyTorch Conv 의 관례."""

    CHANNEL_LAST = "CHANNEL_LAST"
    """(높이, 너비, 채널) — TensorFlow / 대부분의 이미지 라이브러리 관례."""


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """표본 하나의 모양 (배치 축 제외)."""

    shape: tuple[int, ...]
    layout: TensorLayout
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.shape:
            raise InvariantViolation("모양이 비어 있다.", subject="shape")
        if any(dim < 1 for dim in self.shape):
            raise InvariantViolation(
                f"모든 축은 1 이상이어야 한다. (받은 값 {self.shape})", subject="shape"
            )

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def bytes_per_sample(self) -> int:
        """표본 하나가 차지하는 메모리.

        모듈 4(최적화)에서 배치 크기와 입력 크기를 정할 때 이 숫자가 기준이 된다.
        """
        width = {"float32": 4, "float16": 2, "int8": 1, "uint8": 1}.get(self.dtype, 4)
        return self.element_count * width

    def with_batch(self, batch_size: int) -> tuple[int, ...]:
        if batch_size < 1:
            raise InvariantViolation("배치 크기는 1 이상이어야 한다.", subject="batch_size")
        return (batch_size, *self.shape)

    def describe(self) -> str:
        return f"{self.shape} {self.layout.value} {self.dtype}"


@dataclass(frozen=True, slots=True)
class ImageTensorSpec:
    """이미지 입력 명세. (실습 3-3)

    이미지를 숫자로 바꾸는 일은 `크기 조정 → 배열 → 정규화` 세 단계다.
    셋 다 학습과 배포에서 **똑같이** 일어나야 한다. 그래서 계약으로 고정한다.
    """

    width: int
    height: int
    channels: int = 3
    layout: TensorLayout = TensorLayout.CHANNEL_FIRST
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise InvariantViolation("이미지 크기는 1 이상이어야 한다.", subject="size")
        if self.channels not in (1, 3):
            raise InvariantViolation(
                "채널은 1(GRAY) 또는 3(RGB) 만 지원한다.", subject="channels"
            )
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise InvariantViolation(
                f"정규화 통계는 채널 수({self.channels})와 개수가 같아야 한다.",
                subject="mean/std",
            )
        if any(s <= 0 for s in self.std):
            raise InvariantViolation(
                "표준편차가 0 이면 나눌 수 없다.", subject="std"
            )

    def to_tensor_spec(self) -> TensorSpec:
        if self.layout is TensorLayout.CHANNEL_LAST:
            shape = (self.height, self.width, self.channels)
        else:
            shape = (self.channels, self.height, self.width)
        return TensorSpec(shape=shape, layout=self.layout)

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """한 번에 몇 개를 먹일 것인가. (실습 3-1)"""

    sample: TensorSpec
    batch_size: int
    shuffle: bool = True
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise InvariantViolation("배치 크기는 1 이상이어야 한다.", subject="batch_size")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.sample.with_batch(self.batch_size)

    @property
    def bytes_per_batch(self) -> int:
        return self.sample.bytes_per_sample * self.batch_size

    def batch_count(self, sample_count: int) -> int:
        if sample_count < 0:
            raise InvariantViolation("표본 수는 음수일 수 없다.", subject="sample_count")
        full, remainder = divmod(sample_count, self.batch_size)
        return full if (self.drop_last or remainder == 0) else full + 1


@dataclass(frozen=True, slots=True)
class DatasetTensorSummary:
    """데이터를 실제로 텐서로 바꿔 본 결과. (실습 3-1)

    Infrastructure 가 채워 넣는다. 여기에도 배열은 없다 — 모양과 개수와 통계뿐이다.
    """

    split: str
    sample_count: int
    sample_shape: tuple[int, ...]
    class_counts: Mapping[str, int] = field(default_factory=dict)
    feature_min: float | None = None
    feature_max: float | None = None
    feature_mean: float | None = None
    feature_std: float | None = None
    nan_count: int = 0

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise InvariantViolation("표본 수는 음수일 수 없다.", subject="sample_count")
        if self.nan_count < 0:
            raise InvariantViolation("nan_count 는 음수일 수 없다.", subject="nan_count")

    @property
    def class_count(self) -> int:
        return len([c for c in self.class_counts.values() if c > 0])

    @property
    def is_finite(self) -> bool:
        """NaN 이 하나라도 있으면 학습은 첫 배치에서 무너진다."""
        return self.nan_count == 0

    def matches(self, spec: TensorSpec) -> bool:
        return tuple(self.sample_shape) == tuple(spec.shape)

    def describe(self) -> str:
        stats = (
            f"min={self.feature_min:.3f} max={self.feature_max:.3f} "
            f"mean={self.feature_mean:.3f} std={self.feature_std:.3f}"
            if self.feature_mean is not None
            else "통계 없음"
        )
        classes = ", ".join(
            f"{name} {count}" for name, count in sorted(self.class_counts.items())
        )
        return (
            f"{self.split:<11} {self.sample_count:>6,}개  shape={self.sample_shape}\n"
            f"            {stats}\n"
            f"            {classes}"
        )
