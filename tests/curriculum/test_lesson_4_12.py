"""실습 4-12 — 학습 중에 양자화를 가르쳐라.

    pytest -m lesson_4_12 -s

실습 4-7 에서 한 것은 **PTQ**다. 다 배운 모델을 나중에 정수로 눌렀다.
학습 파이프라인을 안 건드려도 되는 것이 장점이다.

문제는 모델이 그 사실을 모르고 배웠다는 것이다.
FP32 로 배운 미세한 차이가 정수 격자에 눌리면서 사라진다.

**QAT** 는 순서를 바꾼다. 배우는 동안 눌러 보고, 눌린 값으로 손실을 계산한다.
그러면 모델이 **눌려도 견디는 가중치**를 찾아간다.

대가는 학습 파이프라인 하나를 영구히 더 유지하는 것이다.
그래서 순서가 있다.

    **PTQ 를 먼저 재 보고, 부족할 때만 QAT 로 간다.**
"""

from __future__ import annotations

import pytest

from application.optimization.compare_quantization import CompareQuantizationCommand
from domain.optimization.quantization import QuantizationSpec, QuantizationApproach
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_4_12


def _compare(optimization_container, run_id, bits, **kwargs):  # noqa: ANN001, ANN003
    return optimization_container.compare_quantization().execute(
        CompareQuantizationCommand(run_id=run_id, bits=bits, epochs=12, **kwargs)
    )


def test_8비트에서는_PTQ_로_충분하다(optimized) -> None:
    report.section("실습 4-12 · 학습 중에 양자화를 가르쳐라")

    view = _compare(optimized.optimization, optimized.run_id, 8)
    report.block("8비트", view.render())

    assert view.post_training.accuracy_drop <= 0.02
    assert "QUANT_PTQ_SUFFICIENT" in [f.code for f in view.findings]
    report.note(
        "손실이 없다. **여기서 QAT 를 도입하면 비용만 는다** — "
        "학습 파이프라인이 하나 더 생기고, 재학습할 때마다 그 절차를 밟아야 한다. "
        "실습 4-7 에서 INT8 정확도가 안 떨어졌던 것과 같은 이야기다."
    )


def test_비트를_좁히면_어디선가_무너진다(optimized) -> None:
    rows = []
    for bits in (8, 4, 3, 2):
        view = _compare(optimized.optimization, optimized.run_id, bits)
        rows.append((bits, view.post_training.quantized_accuracy, view.recovered))

    report.block(
        "비트를 좁혀 가며",
        f"{'비트':>6}{'PTQ 정확도':>14}{'QAT 회복':>12}\n"
        + "\n".join(f"{b:>6}{a:>14.3f}{r:>+12.3f}" for b, a, r in rows),
    )

    assert rows[0][1] >= rows[-1][1]
    report.note(
        "8비트에서는 아무 일도 안 일어난다. **2비트에서 무너진다.** "
        "어디서 무너지는지는 모델마다 다르다 — "
        "그래서 '몇 비트까지 되는가'는 재 봐야 아는 숫자다."
    )


def test_무너지는_지점에서는_QAT_가_되찾는다(optimized) -> None:
    """이 실습의 본론."""
    view = _compare(optimized.optimization, optimized.run_id, 2)
    report.block("2비트", view.render())

    assert view.post_training.accuracy_drop > 0.05
    assert view.recovered > 0.05
    assert "QUANT_QAT_JUSTIFIED" in [f.code for f in view.findings]
    report.note(
        f"PTQ 는 {view.post_training.accuracy_drop:.3f} 를 잃었고 "
        f"QAT 가 {view.recovered:+.3f} 를 되찾았다. "
        "**여기서는 QAT 를 도입할 근거가 있다** — "
        "그리고 그 근거는 정확도가 아니라 '얼마나 되찾는가'다."
    )


def test_QAT_는_공짜가_아니다(optimized) -> None:
    view = _compare(optimized.optimization, optimized.run_id, 2)

    report.block(
        "대가",
        f"  PTQ 학습 시간 : {view.post_training.training_seconds:.2f}s (학습 없음)\n"
        f"  QAT 학습 시간 : {view.quantization_aware.training_seconds:.2f}s\n"
        f"  차이          : {view.extra_training_seconds:+.2f}s",
    )

    assert view.post_training.training_seconds == 0.0
    assert view.quantization_aware.training_seconds > 0.0
    report.note(
        "여기서는 1초 남짓이다. 실제 모델에서는 시간이 배로 든다. "
        "**그리고 그 시간은 한 번이 아니라 재학습할 때마다 든다** (모듈 5, 6). "
        "그래서 '정확도가 조금 더 높다'만으로는 도입 근거가 안 된다."
    )


def test_기울기가_안_흐르면_QAT_는_아무것도_안_한다() -> None:
    """QAT 구현에서 가장 흔한 버그."""
    import torch

    from infrastructure.optimization.quantize_aware import fake_quantize

    spec = QuantizationSpec(
        approach=QuantizationApproach.QUANTIZATION_AWARE, bits=4
    )
    weight = torch.randn(4, 3, 5, requires_grad=True)
    fake_quantize(weight, spec).sum().backward()

    report.block(
        "straight-through estimator",
        f"  round() 의 미분은 거의 모든 점에서 0이다.\n"
        f"  그대로 두면 기울기가 {0.0} 이 되어 아무것도 안 배운다.\n"
        f"  역전파에서 항등함수인 척하면 기울기가 흐른다: "
        f"{float(weight.grad.abs().sum()):.1f}",
    )

    assert weight.grad is not None
    assert float(weight.grad.abs().sum()) > 0
    report.note(
        "**QAT 를 붙였는데 정확도가 그대로라면 이것부터 확인해야 한다.** "
        "가중치에 대입하는 식으로 구현하면 계산 그래프가 끊겨서 "
        "'학습은 도는데 아무것도 안 배우는' 상태가 된다 — 에러는 안 난다."
    )


def test_채널별_배율과_전체_공통_배율은_다르다(optimized) -> None:
    per_channel = _compare(optimized.optimization, optimized.run_id, 3, per_channel=True)
    shared = _compare(optimized.optimization, optimized.run_id, 3, per_channel=False)

    report.block(
        "배율을 어디까지 쪼개는가",
        f"  채널별   PTQ {per_channel.post_training.quantized_accuracy:.3f}\n"
        f"  전체공통 PTQ {shared.post_training.quantized_accuracy:.3f}",
    )

    assert shared.post_training.quantized_accuracy <= (
        per_channel.post_training.quantized_accuracy
    )
    report.note(
        "배율 하나로 전체를 누르면 **가장 큰 채널이 나머지를 다 눌러 버린다.** "
        "채널마다 배율을 따로 잡는 것이 대개 공짜에 가까운 이득이다."
    )


def test_비트_수는_2에서_16_사이여야_한다() -> None:
    with pytest.raises(InvariantViolation, match="2~16"):
        QuantizationSpec(approach=QuantizationApproach.POST_TRAINING, bits=1)
