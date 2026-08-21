"""실습 4-8 — 정확도와 Latency를 직접 싸움 붙여라.

    pytest -m lesson_4_8 -s

여섯 개의 후보가 있다. 각각 무엇을 주고 무엇을 받았는지 한 표에 놓는다.

표를 만들면 대개 하나 이상은 이런 후보가 나온다.
**더 작게 만들었는데 더 느려진 후보.**
"""

from __future__ import annotations

import pytest

from tests.support import report

pytestmark = pytest.mark.lesson_4_8


def test_한_표에_놓아야_교환이_보인다(optimized) -> None:
    report.section("실습 4-8 · 정확도와 Latency를 직접 싸움 붙여라")

    view = optimized.table
    report.block("트레이드오프 표", view.render())

    assert len(view.pareto_front) >= 1
    report.note(
        f"가장 빠른 것은 {view.fastest}, 가장 작은 것은 {view.smallest}, "
        f"가장 정확한 것은 {view.most_accurate} 다. **셋이 서로 다르다.**"
    )
    report.note(
        "'최적'이라는 말은 이 표 앞에서 뜻을 잃는다. "
        "무엇을 기준으로 최적인지 먼저 말해야 한다."
    )


def test_더_작게_만들었는데_더_느려진_후보가_있다(optimized) -> None:
    """양자화한 값을 다시 풀어 쓰는 비용이 계산 이득보다 클 때 그렇다."""
    fp32 = optimized.candidates["TFLITE/FP32"]
    int8 = optimized.candidates["TFLITE/INT8"]

    report.block(
        "TFLite 정밀도별 크기 vs 속도",
        "\n".join(
            [
                f"  {'후보':<14}{'크기':>10}{'p50(ms)':>10}{'p95(ms)':>10}",
                f"  {'FP32':<14}{fp32.size_bytes:>9,}B{fp32.p50_ms:>10.4f}{fp32.p95_ms:>10.4f}",
                f"  {'FP16':<14}{optimized.candidates['TFLITE/FP16'].size_bytes:>9,}B"
                f"{optimized.candidates['TFLITE/FP16'].p50_ms:>10.4f}"
                f"{optimized.candidates['TFLITE/FP16'].p95_ms:>10.4f}",
                f"  {'INT8':<14}{int8.size_bytes:>9,}B{int8.p50_ms:>10.4f}{int8.p95_ms:>10.4f}",
            ]
        ),
    )

    assert int8.size_bytes < fp32.size_bytes
    assert int8.p95_ms > fp32.p95_ms
    report.note(
        f"INT8 이 FP32 보다 작다({int8.size_bytes:,} < {fp32.size_bytes:,}). "
        f"그런데 **더 느리다**({int8.p95_ms:.4f} > {fp32.p95_ms:.4f})."
    )
    report.note(
        "이 CPU 에는 정수 연산을 몰아 처리하는 유닛이 없다. "
        "그래서 INT8 로 저장한 값을 매번 실수로 되돌린 뒤 계산한다 — 그 비용이 더 크다."
    )
    report.note(
        "'양자화하면 빨라진다'는 문장에는 **하드웨어가 그것을 지원할 때**가 빠져 있다."
    )


def test_지배당한_후보는_볼_필요가_없다(optimized) -> None:
    """다른 후보에게 모든 면에서 지는 후보는 검토 대상에서 빠진다."""
    view = optimized.table
    report.block(
        "파레토 전선",
        "\n".join(
            [
                f"  전선에 남은 후보 : {', '.join(view.pareto_front)}",
                f"  기준보다 느린 후보 : {', '.join(view.slower_than_baseline) or '없음'}",
            ]
        ),
    )
    assert "PYTORCH/FP32" not in view.pareto_front
    report.note(
        "기준 모델(PYTORCH/FP32)은 전선에서 빠졌다. "
        "더 빠르면서 정확도가 같은 후보가 있기 때문이다."
    )


def test_속도만_보면_잘못_고른다(optimized) -> None:
    """이 표에서 '가장 빠른 것'을 고르는 것이 왜 답이 아닌지. (실습 4-10 으로 이어진다)"""
    view = optimized.table
    fastest = optimized.candidates[view.fastest]
    slowest_deployable = max(
        (c for c in optimized.candidates.values()), key=lambda c: c.p95_ms
    )

    report.block(
        "가장 빠른 후보 vs 가장 느린 후보",
        "\n".join(
            [
                f"  {fastest.label:<18} p95 {fastest.p95_ms:.4f}ms  "
                f"macro recall {fastest.macro_recall:.4f}",
                f"  {slowest_deployable.label:<18} p95 {slowest_deployable.p95_ms:.4f}ms  "
                f"macro recall {slowest_deployable.macro_recall:.4f}",
            ]
        ),
    )
    assert fastest.p95_ms < slowest_deployable.p95_ms
    report.note(
        f"두 후보의 차이는 {slowest_deployable.p95_ms - fastest.p95_ms:.4f}ms 다. "
        "설비 사이클이 30ms 라면 이 차이는 아무 값어치가 없다."
    )
    report.note(
        "예산을 이미 만족한 뒤에 남은 여유는 성능이 아니다. "
        "그 여유를 얻으려고 정확도를 내줬다면 그것은 손해다."
    )


def test_같은_조합은_하나만_남는다(optimization_container, optimized) -> None:
    """다시 변환하면 덮어쓴다. 같은 (경로, 정밀도)가 두 줄로 남으면 표가 거짓말을 한다."""
    from domain.optimization.runtime import Precision, RuntimeTarget
    from tests.support import optimization_scenario as os4

    os4.start(
        optimization_container,
        run_id="opt-dup",
        training_run_id=optimized.training_run_id,
    )
    os4.benchmark(optimization_container, "opt-dup")
    os4.convert(optimization_container, "opt-dup", RuntimeTarget.ONNX, Precision.FP32)
    os4.convert(optimization_container, "opt-dup", RuntimeTarget.ONNX, Precision.FP32)

    table = os4.compare(optimization_container, "opt-dup")
    labels = [line for line in table.table.splitlines() if "ONNX/FP32" in line]
    assert len(labels) == 1
    report.note("두 번 변환했지만 표에는 한 줄이다.")
