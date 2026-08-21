"""실습 3-3 — 이미지를 숫자로 바꿔라.

    pytest -m lesson_3_3 -s

이미지를 숫자로 바꾸는 일은 세 단계다.

    1. 크기 조정   — 모델은 고정된 크기만 받는다
    2. 배열 변환   — (H, W, C) uint8 → (C, H, W) float32
    3. 정규화      — 채널마다 평균을 빼고 표준편차로 나눈다

셋 다 **학습과 배포에서 똑같아야** 한다. 하나라도 다르면 현장에서만 틀린다.
"""

from __future__ import annotations

import pytest

from application.model.dto import TensorSummaryView
from domain.model.architecture import ArchitectureKind, ModelArchitecture
from domain.model.tensor_spec import ImageTensorSpec, TensorLayout
from domain.shared.errors import InvariantViolation
from infrastructure.ml.torch_architecture import TorchArchitectureProfiler
from infrastructure.ml.torch_materializer import PillowImageTensorMaterializer
from tests.support import report

pytestmark = pytest.mark.lesson_3_3


def test_이미지가_텐서가_되는_순간(castings) -> None:
    report.section("실습 3-3 · 이미지를 숫자로 바꿔라")

    spec = ImageTensorSpec(width=64, height=64, channels=3)
    summaries = PillowImageTensorMaterializer().materialize(
        str(castings.root), spec
    )
    view = TensorSummaryView.of(summaries["all"])

    report.block("이미지 → 텐서", view.render())

    assert view.sample_shape == (3, 64, 64)
    assert set(view.class_counts) == {"ok", "ng"}
    report.note(
        "96×96 과 128×128 이 섞여 있던 이미지가 전부 (3, 64, 64) 가 되었다. "
        "모델은 고정된 크기만 받기 때문이다."
    )


def test_깨진_파일은_조용히_건너뛰지_않고_세어_둔다(castings) -> None:
    """실습 1-4 에서 찾았던 그 3장이다."""
    summaries = PillowImageTensorMaterializer().materialize(
        str(castings.root), ImageTensorSpec(width=64, height=64)
    )
    assert summaries["unreadable"].sample_count == 3
    report.note(
        "51장 중 3장은 열리지 않는다. "
        "여기서 세어 두지 않으면 '48장으로 학습했다'는 사실이 어디에도 남지 않는다."
    )


def test_정규화가_실제로_값을_옮겼는지_본다(castings) -> None:
    raw = PillowImageTensorMaterializer().materialize(
        str(castings.root),
        ImageTensorSpec(
            width=64, height=64, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    )["all"]
    normalized = PillowImageTensorMaterializer().materialize(
        str(castings.root), ImageTensorSpec(width=64, height=64)
    )["all"]

    report.block(
        "정규화 전후",
        f"  전 : min={raw.feature_min:6.3f} max={raw.feature_max:6.3f} "
        f"mean={raw.feature_mean:6.3f}\n"
        f"  후 : min={normalized.feature_min:6.3f} max={normalized.feature_max:6.3f} "
        f"mean={normalized.feature_mean:6.3f}",
    )
    assert 0.0 <= raw.feature_min and raw.feature_max <= 1.0
    assert normalized.feature_min < 0.0
    report.note(
        "0~1 이던 값이 0 근처로 옮겨졌다. "
        "이 mean/std 는 배포할 때도 **똑같은 값**을 써야 한다."
    )


def test_축_순서를_틀리면_모델이_다른_것을_본다() -> None:
    """PyTorch 는 (C, H, W), 대부분의 이미지 라이브러리는 (H, W, C) 다."""
    torch_style = ImageTensorSpec(
        width=224, height=224, layout=TensorLayout.CHANNEL_FIRST
    ).to_tensor_spec()
    tf_style = ImageTensorSpec(
        width=224, height=224, layout=TensorLayout.CHANNEL_LAST
    ).to_tensor_spec()

    assert torch_style.shape == (3, 224, 224)
    assert tf_style.shape == (224, 224, 3)
    assert torch_style.element_count == tf_style.element_count

    report.block(
        "같은 이미지, 다른 축 순서",
        f"  PyTorch    : {torch_style.shape}\n"
        f"  TensorFlow : {tf_style.shape}\n"
        f"  원소 수는 둘 다 {torch_style.element_count:,}개로 같다.",
    )
    report.note(
        "원소 수가 같아서 reshape 이 그냥 된다. "
        "그래서 축을 틀려도 에러가 안 나고, 모델은 조용히 이상한 것을 배운다."
    )


def test_정규화_통계는_채널_수와_맞아야_한다() -> None:
    with pytest.raises(InvariantViolation, match="채널 수"):
        ImageTensorSpec(width=64, height=64, channels=1, mean=(0.5, 0.5, 0.5))
    with pytest.raises(InvariantViolation, match="표준편차가 0"):
        ImageTensorSpec(width=64, height=64, std=(0.0, 0.2, 0.2))


def test_이미지_모델의_계산량은_입력_크기의_제곱으로_는다() -> None:
    """모듈 4 에서 입력 크기를 줄이는 이유가 여기 있다."""
    profiler = TorchArchitectureProfiler()

    def macs(size: int) -> int:
        spec = ImageTensorSpec(width=size, height=size).to_tensor_spec()
        return profiler.profile(
            ModelArchitecture(
                kind=ArchitectureKind.CNN2D,
                input_spec=spec,
                class_count=2,
                hidden_channels=(8, 16),
                kernel_size=3,
            )
        ).mac_count

    small, large = macs(64), macs(128)
    report.block(
        "입력 크기와 연산량",
        f"  64×64   : {small:>12,} MACs\n"
        f"  128×128 : {large:>12,} MACs  ({large / small:.1f}배)",
    )
    assert large > small * 3
    report.note("한 변을 2배로 하면 연산은 약 4배가 된다. 정확도는 4배가 되지 않는다.")
