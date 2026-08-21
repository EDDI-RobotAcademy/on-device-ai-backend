"""실습 3-2 — AI가 데이터를 계산하는 과정을 직접 뜯어보자.

    pytest -m lesson_3_2 -s

층마다 무엇이 들어가서 무엇이 나오는지, 그리고 그 사이에 곱셈이 몇 번 일어나는지.
이 두 숫자가 모듈 4(최적화) 전체의 출발점이다.
"""

from __future__ import annotations

import pytest

from application.model.dto import ArchitectureProfileView
from domain.model.architecture import ArchitectureKind, ModelArchitecture
from domain.model.tensor_spec import TensorLayout, TensorSpec
from domain.shared.errors import InvariantViolation
from infrastructure.ml.torch_architecture import TorchArchitectureProfiler
from tests.support import model_scenario as ms
from tests.support import report

pytestmark = pytest.mark.lesson_3_2


def profile_of(architecture: ModelArchitecture) -> ArchitectureProfileView:
    return ArchitectureProfileView.of(
        "-", TorchArchitectureProfiler().profile(architecture)
    )


def test_층마다_무엇이_나오는지_직접_본다() -> None:
    report.section("실습 3-2 · AI가 데이터를 계산하는 과정을 직접 뜯어보자")

    view = profile_of(ms.cnn_architecture())
    report.block("CNN1D 층별 계산", view.render())

    assert view.parameter_count > 0
    assert view.mac_count > 0
    report.note(
        f"파라미터 {view.parameter_count:,}개 "
        f"({view.parameter_bytes / 1024:.1f} KiB) / "
        f"추론 1회 {view.mac_count / 1e6:.2f} MMACs"
    )


def test_가장_무거운_층과_가장_바쁜_층은_다르다() -> None:
    """이 구분을 모르면 최적화를 엉뚱한 곳에 한다. (모듈 4 로 이어진다)"""
    view = profile_of(ms.cnn_architecture(hidden=(16, 32)))

    report.block(
        "무게 vs 연산량",
        f"  가장 무거운 층(파라미터) : {view.heaviest_layer}\n"
        f"  가장 바쁜 층(MAC)        : {view.busiest_layer}",
    )
    assert view.heaviest_layer is not None
    assert view.busiest_layer is not None
    report.note(
        "Conv 층은 파라미터가 적어도 연산이 많다 — 커널을 모든 위치에서 다시 쓰기 때문이다. "
        "메모리를 줄이려면 무거운 층을, 속도를 올리려면 바쁜 층을 봐야 한다."
    )


def test_같은_입력_다른_구조_다른_비용() -> None:
    """MLP 는 시간 축을 펼쳐 버린다. 파라미터는 폭증하고 순서는 못 본다."""
    cnn = profile_of(ms.cnn_architecture())
    mlp = profile_of(ms.mlp_architecture())

    report.block(
        "같은 (30, 6) 입력, 두 구조",
        f"{'':12}{'params':>12}{'MACs':>14}{'FP32':>12}\n"
        f"{'CNN1D':12}{cnn.parameter_count:>12,}{cnn.mac_count:>14,}"
        f"{cnn.parameter_bytes / 1024:>10.1f}K\n"
        f"{'MLP':12}{mlp.parameter_count:>12,}{mlp.mac_count:>14,}"
        f"{mlp.parameter_bytes / 1024:>10.1f}K",
    )
    assert mlp.parameter_count > cnn.parameter_count
    report.note(
        "MLP 는 (30, 6) 을 180 개짜리 벡터로 펴서 첫 층에 통째로 연결한다. "
        "파라미터가 훨씬 많은데도 시간 순서는 보지 못한다."
    )


def test_은닉_채널을_두_배로_하면_비용이_어떻게_변하는가() -> None:
    small = profile_of(ms.cnn_architecture(hidden=(16, 32)))
    large = profile_of(ms.cnn_architecture(hidden=(32, 64)))

    report.block(
        "채널 2배",
        f"  파라미터 {small.parameter_count:,} → {large.parameter_count:,} "
        f"({large.parameter_count / small.parameter_count:.1f}배)\n"
        f"  MAC      {small.mac_count:,} → {large.mac_count:,} "
        f"({large.mac_count / small.mac_count:.1f}배)",
    )
    assert large.parameter_count > small.parameter_count * 3
    report.note("채널을 2배로 하면 Conv 파라미터는 대략 4배가 된다. 선형이 아니다.")


def test_구조_명세는_PyTorch_를_모른다() -> None:
    """Domain 은 '어떤 모양의 계산을 할 것인가'만 안다."""
    architecture = ms.cnn_architecture()
    assert architecture.describe().startswith("CNN1D")

    with pytest.raises(InvariantViolation, match="차원 입력을 기대"):
        ModelArchitecture(
            kind=ArchitectureKind.CNN2D,  # 이미지용인데
            input_spec=TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST),
            class_count=3,
        )

    with pytest.raises(InvariantViolation, match="홀수"):
        ms.cnn_architecture()
        ModelArchitecture(
            kind=ArchitectureKind.CNN1D,
            input_spec=ms.input_spec(),
            class_count=3,
            kernel_size=4,
        )

    report.note(
        "이 파일들에 torch 라는 단어는 없다. "
        "nn.Module 로 조립하는 일은 infrastructure/ml 에서만 일어난다."
    )


def test_학습_준비에_구조_프로파일이_함께_기록된다(trained) -> None:
    from application.model.support import load_run

    run = load_run(trained.model.runs, trained.run_id)
    assert run.profile is not None
    assert run.profile.parameter_count > 0
    report.note(
        "학습을 시작하기 전에 이 숫자가 기록된다. "
        "나중에 '왜 이 모델이 디바이스에 안 들어가는가'를 되짚을 근거가 된다."
    )
