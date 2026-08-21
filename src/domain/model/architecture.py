"""모델의 구조와 그 계산량. (실습 3-2)

Domain 은 "어떤 모양의 계산을 할 것인가"만 안다.
그 계산을 실제로 하는 것은 PyTorch 고, 그건 Infrastructure 다. (CLAUDE.md §14)

여기서 세는 두 숫자가 모듈 4(최적화) 전체의 출발점이 된다.

    parameter_count  모델이 들고 다녀야 하는 무게 → 메모리, 저장 공간
    mac_count        추론 한 번에 필요한 곱셈-덧셈 수 → 지연시간

"PC에서 돌아간다"와 "디바이스에서 30ms 안에 돌아간다" 사이의 거리가 이 숫자에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.model.tensor_spec import TensorSpec
from domain.shared.errors import InvariantViolation


class ArchitectureKind(Enum):
    MLP = "MLP"
    """완전연결. 시간 축을 펼쳐 버리므로 순서를 못 본다."""

    CNN1D = "CNN1D"
    """1차원 합성곱. 시계열의 국소 패턴을 본다."""

    CNN2D = "CNN2D"
    """2차원 합성곱. 이미지."""


class GlobalPooling(Enum):
    """마지막에 공간 축을 어떻게 접을 것인가. (실습 3-11)

    이 한 줄이 **작은 결함을 볼 수 있는지**를 정한다.

        AVERAGE  전체를 평균낸다. 넓게 퍼진 변화에 강하다.
                 그러나 몇 화소짜리 이물은 평균에 묻혀 사라진다.
        MAX      가장 강하게 반응한 한 곳만 남긴다.
                 작고 국소적인 결함에 강하다. 대신 잡음 한 점에도 반응한다.

    "어느 쪽이 옳은가"는 코드가 아니라 **결함의 생김새**가 정한다.
    """

    AVERAGE = "AVERAGE"
    MAX = "MAX"


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """어떤 모양의 계산을 할 것인가.

    PyTorch 의 nn.Module 이 아니다. 그것을 **만들기 위한 명세**다.
    Infrastructure 의 어댑터가 이 명세를 읽어 실제 신경망을 조립한다.
    """

    kind: ArchitectureKind
    input_spec: TensorSpec
    class_count: int
    hidden_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 5
    dropout: float = 0.1
    pooling: GlobalPooling = GlobalPooling.AVERAGE
    """공간 축을 접는 방식. (실습 3-11)"""

    def __post_init__(self) -> None:
        if self.class_count < 2:
            raise InvariantViolation(
                "클래스가 둘 미만이면 분류 문제가 아니다.", subject="class_count"
            )
        if not self.hidden_channels:
            raise InvariantViolation("은닉층이 하나도 없다.", subject="hidden_channels")
        if any(c < 1 for c in self.hidden_channels):
            raise InvariantViolation(
                "은닉 채널 수는 1 이상이어야 한다.", subject="hidden_channels"
            )
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise InvariantViolation(
                "커널 크기는 1 이상의 홀수여야 한다.", subject="kernel_size"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise InvariantViolation("dropout 은 0 이상 1 미만이어야 한다.", subject="dropout")

        rank = len(self.input_spec.shape)
        expected = {
            ArchitectureKind.MLP: (1, 2),
            ArchitectureKind.CNN1D: (2,),
            ArchitectureKind.CNN2D: (3,),
        }[self.kind]
        if rank not in expected:
            raise InvariantViolation(
                f"{self.kind.value} 는 {expected} 차원 입력을 기대하는데 "
                f"{rank} 차원이 들어왔다.",
                subject="input_spec",
            )

    @property
    def depth(self) -> int:
        return len(self.hidden_channels)

    @property
    def width(self) -> int:
        """가장 넓은 층의 채널 수. 구조를 줄일 때 이 숫자를 건드린다. (실습 4-11)"""
        return max(self.hidden_channels)

    def with_channels(self, hidden_channels: tuple[int, ...]) -> ModelArchitecture:
        """채널 수만 바꾼 같은 구조. (실습 4-11)

        구조를 줄이는 것은 **다시 학습해야 하는 일**이다.
        숫자를 줄이는 양자화와 여기서 갈린다.
        """
        return ModelArchitecture(
            kind=self.kind,
            input_spec=self.input_spec,
            class_count=self.class_count,
            hidden_channels=hidden_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            pooling=self.pooling,
        )

    def with_pooling(self, pooling: GlobalPooling) -> ModelArchitecture:
        """접는 방식만 바꾼 같은 구조. (실습 3-11)"""
        return ModelArchitecture(
            kind=self.kind,
            input_spec=self.input_spec,
            class_count=self.class_count,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            pooling=pooling,
        )

    def describe(self) -> str:
        return (
            f"{self.kind.value} input={self.input_spec.shape} "
            f"hidden={self.hidden_channels} k={self.kernel_size} "
            f"pool={self.pooling.value} → {self.class_count} classes"
        )


@dataclass(frozen=True, slots=True)
class LayerProfile:
    """층 하나가 무엇을 하는가."""

    name: str
    kind: str
    output_shape: tuple[int, ...]
    parameter_count: int
    mac_count: int = 0
    """Multiply-Accumulate. 곱셈 한 번 + 덧셈 한 번을 1로 센다."""

    def __post_init__(self) -> None:
        if self.parameter_count < 0 or self.mac_count < 0:
            raise InvariantViolation("개수는 음수일 수 없다.", subject=self.name)


@dataclass(frozen=True, slots=True)
class ArchitectureProfile:
    """구조를 실제로 조립해 본 결과. Infrastructure 가 채운다. (실습 3-2)"""

    layers: tuple[LayerProfile, ...] = field(default_factory=tuple)
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()

    @property
    def parameter_count(self) -> int:
        return sum(layer.parameter_count for layer in self.layers)

    @property
    def mac_count(self) -> int:
        return sum(layer.mac_count for layer in self.layers)

    @property
    def parameter_bytes(self) -> int:
        """FP32 기준 무게. 모듈 4 에서 이 숫자를 반으로, 사분의 일로 줄인다."""
        return self.parameter_count * 4

    @property
    def heaviest_layer(self) -> LayerProfile | None:
        if not self.layers:
            return None
        return max(self.layers, key=lambda layer: layer.parameter_count)

    @property
    def busiest_layer(self) -> LayerProfile | None:
        """연산이 가장 많은 층. 무거운 층과 다를 수 있다 — 그게 요점이다."""
        if not self.layers:
            return None
        return max(self.layers, key=lambda layer: layer.mac_count)

    def render(self) -> str:
        lines = [
            f"{'layer':<20}{'kind':<12}{'output':<22}{'params':>12}{'MACs':>14}",
            "-" * 80,
        ]
        for layer in self.layers:
            lines.append(
                f"{layer.name:<20}{layer.kind:<12}{str(layer.output_shape):<22}"
                f"{layer.parameter_count:>12,}{layer.mac_count:>14,}"
            )
        lines.append("-" * 80)
        lines.append(
            f"{'합계':<54}{self.parameter_count:>12,}{self.mac_count:>14,}"
        )
        lines.append(
            f"  FP32 무게 {self.parameter_bytes / 1024:.1f} KiB "
            f"/ 추론 1회 {self.mac_count / 1e6:.2f} MMACs"
        )
        return "\n".join(lines)
