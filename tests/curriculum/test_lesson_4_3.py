"""실습 4-3 — ONNX로 모델의 실행 경로를 바꿔라.

    pytest -m lesson_4_3 -s

ONNX 는 파일 형식이자 **연산자 규격**이다.
PyTorch 의 연산이 ONNX 연산자로 치환되므로, 그 과정에서 의미가 조금 바뀔 수 있다.

그래서 여기서 확인할 것은 두 가지다.
    1. 프레임워크를 벗어났는가
    2. 벗어나면서 답이 바뀌지 않았는가
"""

from __future__ import annotations

import pytest

from tests.support import report

pytestmark = pytest.mark.lesson_4_3


def test_프레임워크_중립_그래프로_나왔다(optimized) -> None:
    report.section("실습 4-3 · ONNX로 모델의 실행 경로를 바꿔라")

    view = optimized.candidates["ONNX/FP32"]
    report.block("ONNX 변환", view.render())

    assert view.runtime == "ONNX"
    assert view.size_bytes > 0
    report.note(
        "이 파일은 PyTorch 를 요구하지 않는다. "
        "ONNX Runtime, TensorRT, OpenVINO 등이 같은 파일을 읽는다."
    )


def test_연산자_치환은_숫자를_아주_조금_바꾼다(optimized) -> None:
    """0 이 아니다. 그런데 답은 바뀌지 않았다."""
    view = optimized.candidates["ONNX/FP32"]
    torchscript = optimized.candidates["TORCHSCRIPT/FP32"]

    report.block("두 변환의 동등성 비교", "\n".join([
        f"  TORCHSCRIPT : {torchscript.equivalence}",
        f"  ONNX        : {view.equivalence}",
    ]))
    report.note(
        "TorchScript 는 같은 커널을 쓰므로 차이가 정확히 0 이다. "
        "ONNX 는 다른 구현을 쓰므로 부동소수점 마지막 자리가 흔들린다."
    )
    report.note("중요한 것은 그 흔들림이 예측을 바꾸지 않았다는 사실이다.")
    assert "argmax 일치 100.00%" in view.equivalence
    # 동등성 쪽 소견이 하나도 없다.
    # (벤치마크 소견은 기계가 바쁘면 붙을 수 있다 — 그건 변환의 문제가 아니다)
    assert [f.code for f in view.findings if f.code.startswith("CONVERT_")] == []


def test_ONNX_는_오버헤드가_가장_작다(optimized) -> None:
    """같은 가중치를 담는데 파일 크기는 형식마다 다르다. (실습 4-5 로 이어진다)"""
    rows = sorted(
        (v for v in optimized.candidates.values()),
        key=lambda v: v.overhead_bytes,
    )
    report.block(
        "형식별 오버헤드",
        "\n".join(
            f"  {v.label:<18} 가중치 {v.theoretical_weight_bytes:>7,}B "
            f"+ 오버헤드 {v.overhead_bytes:>7,}B"
            for v in rows
        ),
    )
    onnx = optimized.candidates["ONNX/FP32"]
    torchscript = optimized.candidates["TORCHSCRIPT/FP32"]
    assert onnx.overhead_bytes < torchscript.overhead_bytes
    report.note(
        "TorchScript 는 파이썬 코드 조각과 디버깅 정보를 함께 담는다. "
        "ONNX 는 그래프만 담는다."
    )


def test_배치_축만_동적으로_열려_있다(optimized) -> None:
    """나머지 축까지 동적이면 런타임이 최적화를 포기한다."""
    import numpy as np

    from domain.optimization.identifiers import ArtifactId

    view = optimized.candidates["ONNX/FP32"]
    loaded = optimized.optimization.runtimes.require(ArtifactId.of(view.artifact_id))

    one = loaded.predict(np.zeros((1, *loaded.input_shape), dtype="float32"))
    eight = loaded.predict(np.zeros((8, *loaded.input_shape), dtype="float32"))

    assert one.shape[0] == 1
    assert eight.shape[0] == 8
    report.note(
        "배치만 바뀐다. 창 길이(30)와 채널 수(6)는 고정이다 — "
        "그 두 개가 열려 있으면 런타임이 메모리 계획을 세울 수 없다."
    )
