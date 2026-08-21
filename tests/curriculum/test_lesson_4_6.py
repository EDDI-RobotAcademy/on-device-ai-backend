"""실습 4-6 — FP16으로 줄이고 무엇을 얻었는지 확인하라.

    pytest -m lesson_4_6 -s

FP16 은 숫자 하나를 4바이트 대신 2바이트로 쓴다.
그러니 파일이 절반이 되어야 한다 — **가 아니다.**

여기서 실제로 얻은 것과 잃은 것을 각각 센다.
"""

from __future__ import annotations

import pytest

from tests.support import report

pytestmark = pytest.mark.lesson_4_6

PARAMETER_COUNT = 3187


def test_가중치는_절반이_되지만_파일은_절반이_되지_않는다(optimized) -> None:
    report.section("실습 4-6 · FP16으로 줄이고 무엇을 얻었는지 확인하라")

    fp32 = optimized.candidates["TFLITE/FP32"]
    fp16 = optimized.candidates["TFLITE/FP16"]

    report.block(
        "FP32 → FP16",
        "\n".join(
            [
                f"  FP32  파일 {fp32.size_bytes:>7,}B = 가중치 {fp32.theoretical_weight_bytes:>7,}"
                f" + 오버헤드 {fp32.overhead_bytes:>7,}",
                f"  FP16  파일 {fp16.size_bytes:>7,}B = 가중치 {fp16.theoretical_weight_bytes:>7,}"
                f" + 오버헤드 {fp16.overhead_bytes:>7,}",
            ]
        ),
    )

    # 이론: 가중치는 정확히 절반이다.
    assert fp16.theoretical_weight_bytes == fp32.theoretical_weight_bytes // 2
    # 실제: 파일은 절반보다 크다.
    ratio = fp16.size_bytes / fp32.size_bytes
    assert 0.5 < ratio < 1.0

    report.note(
        f"가중치는 정확히 절반(12,748 → 6,374). "
        f"그런데 파일은 {ratio:.0%} 로만 줄었다."
    )
    report.note(
        f"오버헤드가 {fp32.overhead_bytes:,}B → {fp16.overhead_bytes:,}B 로 "
        "오히려 늘었기 때문이다 — 양자화 파라미터가 함께 들어간다."
    )


def test_소수점_아래가_흔들리지만_답은_그대로다(optimized) -> None:
    fp32 = optimized.candidates["TFLITE/FP32"]
    fp16 = optimized.candidates["TFLITE/FP16"]

    report.block(
        "정밀도별 출력 차이",
        "\n".join([f"  FP32 : {fp32.equivalence}", f"  FP16 : {fp16.equivalence}"]),
    )
    assert "argmax 일치 100.00%" in fp16.equivalence
    assert fp16.accuracy == pytest.approx(fp32.accuracy)
    report.note(
        "max|diff| 가 1e-06 대에서 1e-03 대로 커졌다. 값은 분명히 흔들렸다. "
        "그런데 어느 클래스가 가장 큰지는 바뀌지 않았다."
    )
    report.note(
        "FP16 을 FP32 기준(1e-4)으로 재면 전부 실패한다. "
        "그래서 EquivalencePolicy 는 정밀도마다 다른 허용 오차를 쓴다."
    )


def test_FP16은_속도를_보장하지_않는다(optimized) -> None:
    """줄인 것은 저장 공간이다. 속도는 실행기가 그 형식을 지원해야 빨라진다."""
    fp32 = optimized.candidates["TFLITE/FP32"]
    fp16 = optimized.candidates["TFLITE/FP16"]

    report.block(
        "속도 비교",
        "\n".join(
            [
                f"  TFLITE/FP32  p50 {fp32.p50_ms:.4f}ms  p95 {fp32.p95_ms:.4f}ms",
                f"  TFLITE/FP16  p50 {fp16.p50_ms:.4f}ms  p95 {fp16.p95_ms:.4f}ms",
            ]
        ),
    )
    report.note(
        "이 CPU 에는 FP16 연산기가 없다. 그래서 인터프리터는 "
        "**FP16 으로 저장된 가중치를 FP32 로 풀어서** 계산한다."
    )
    report.note(
        "즉 얻은 것은 파일 크기와 메모리 대역폭이고, 연산 속도는 거의 그대로다. "
        "디바이스에 FP16 유닛이 있으면 그때 속도도 따라온다."
    )
    assert fp16.p95_ms > 0


def test_허용_오차는_정밀도마다_다르다() -> None:
    from domain.optimization.conversion import EquivalencePolicy
    from domain.optimization.runtime import Precision

    policy = EquivalencePolicy()
    assert policy.tolerance_for(Precision.FP32) == 1e-4
    assert policy.tolerance_for(Precision.FP16) == 1e-2
    assert policy.tolerance_for(Precision.INT8) is None

    report.note(
        "INT8 은 허용 오차가 없다 — 값 자체를 비교하지 않는다. "
        "애초에 다른 숫자 체계를 쓰기 때문이다. (실습 4-7)"
    )
