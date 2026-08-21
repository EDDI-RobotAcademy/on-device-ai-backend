"""실행 경로와 정밀도. (실습 4-2 ~ 4-7)

같은 모델을 디바이스에 넣는 방법은 두 축으로 갈린다.

    실행 경로  누가 그 계산을 실행하는가
               PyTorch(파이썬) → TorchScript → ONNX Runtime → TFLite

    정밀도     숫자 하나를 몇 바이트로 표현하는가
               FP32(4) → FP16(2) → INT8(1)

이 파일에는 torch 도 tensorflow 도 없다. **바이트 수와 이름뿐이다.**
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.optimization.identifiers import ArtifactId
from domain.shared.errors import InvariantViolation


class RuntimeTarget(Enum):
    PYTORCH = "PYTORCH"
    """파이썬 인터프리터가 층마다 파이썬 함수를 부른다. 개발에는 좋고 배포에는 무겁다."""

    TORCHSCRIPT = "TORCHSCRIPT"
    """파이썬을 벗어난 그래프. 파이썬 없이 실행되지만 여전히 PyTorch 런타임이 필요하다."""

    ONNX = "ONNX"
    """프레임워크 중립 그래프. 여러 런타임이 같은 파일을 읽는다."""

    TFLITE = "TFLITE"
    """임베디드용. 인터프리터가 작고, 연산자가 제한되어 있다."""

    @property
    def is_deployable_on_device(self) -> bool:
        """디바이스에 실제로 올릴 수 있는가.

        PyTorch 는 파이썬을 요구한다. 대부분의 임베디드 환경에 파이썬은 없다.
        """
        return self is not RuntimeTarget.PYTORCH


class Precision(Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"

    @property
    def bytes_per_weight(self) -> int:
        return {"FP32": 4, "FP16": 2, "INT8": 1}[self.value]

    @property
    def is_quantized(self) -> bool:
        """정수로 바꿨는가. 바꿨다면 정확도 손실을 반드시 확인해야 한다."""
        return self is Precision.INT8


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """변환 결과물 하나.

    크기는 **파일에서 실제로 잰 값**이다. 계산한 값이 아니다.
    그 둘의 차이가 실습 4-5 의 요점이다.
    """

    artifact_id: ArtifactId
    runtime: RuntimeTarget
    precision: Precision
    size_bytes: int
    uri: str
    parameter_count: int = 0

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise InvariantViolation("크기는 음수일 수 없다.", subject="size_bytes")
        if self.parameter_count < 0:
            raise InvariantViolation(
                "파라미터 수는 음수일 수 없다.", subject="parameter_count"
            )
        if not self.uri.strip():
            raise InvariantViolation("결과물 위치가 없다.", subject="uri")

    @property
    def label(self) -> str:
        return f"{self.runtime.value}/{self.precision.value}"

    @property
    def size_kib(self) -> float:
        return self.size_bytes / 1024

    @property
    def theoretical_weight_bytes(self) -> int:
        """파라미터만 담았을 때의 크기.

        3,187개 × 4바이트 = 12,748바이트. 이론상 이만큼이면 된다.
        """
        return self.parameter_count * self.precision.bytes_per_weight

    @property
    def overhead_bytes(self) -> int:
        """가중치 외의 것 — 그래프 구조, 연산자 목록, 메타데이터.

        작은 모델일수록 이 몫이 크다. 그래서 양자화 이득이 이론만큼 나오지 않는다.
        """
        return max(0, self.size_bytes - self.theoretical_weight_bytes)

    @property
    def overhead_ratio(self) -> float:
        if self.size_bytes == 0:
            return 0.0
        return self.overhead_bytes / self.size_bytes

    def size_ratio_to(self, other: ModelArtifact) -> float:
        """other 대비 몇 배인가. 1보다 작으면 줄어든 것이다."""
        if other.size_bytes == 0:
            return 0.0
        return self.size_bytes / other.size_bytes

    def describe(self) -> str:
        return (
            f"{self.label:<18}{self.size_bytes:>9,}B "
            f"({self.size_kib:>7.1f} KiB)  "
            f"가중치 {self.theoretical_weight_bytes:>7,}B "
            f"+ 오버헤드 {self.overhead_bytes:>7,}B ({self.overhead_ratio:.0%})"
        )
