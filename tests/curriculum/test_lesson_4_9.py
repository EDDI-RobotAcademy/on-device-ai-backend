"""실습 4-9 — Memory와 Compute 병목을 직접 찾아라.

    pytest -m lesson_4_9 -s

층이 느린 이유는 둘 중 하나다.

    계산이 많다      → 연산을 줄이면 빨라진다
    데이터를 나른다  → **연산을 줄여도 안 빨라진다**

가르는 기준은 산술 강도(arithmetic intensity) — 1바이트를 나를 때 몇 번 계산하는가.
그리고 기계마다 균형점이 있다. 그 균형점을 모르면 최적화를 엉뚱한 곳에 한다.
"""

from __future__ import annotations

import pytest

from domain.optimization.roofline import (
    BottleneckKind,
    DeviceCapability,
    LayerCost,
    RooflinePolicy,
    RooflineProfile,
)
from tests.support import report

pytestmark = pytest.mark.lesson_4_9


def test_층마다_병목이_다르다(optimized) -> None:
    report.section("실습 4-9 · Memory와 Compute 병목을 직접 찾아라")

    view = optimized.roofline
    report.block("층별 비용 구조", view.render())

    assert view.total_macs > 0
    assert view.machine_balance > 0
    report.note(
        f"이 보드의 균형점은 {view.machine_balance:.1f} MAC/byte 다. "
        "산술 강도가 이보다 낮은 층은 메모리가 못 따라오고 있다는 뜻이다."
    )
    report.note(
        f"연산의 대부분을 차지하는 병목은 {view.dominant_bottleneck}, "
        f"가장 바쁜 층은 {view.busiest_layer} 다."
    )


def test_한_층이_전체_연산을_차지하면_거기부터_고친다(optimized) -> None:
    view = optimized.roofline
    codes = {f.code for f in view.findings}

    report.block("소견", "\n".join(f"  - {f.describe()}" for f in view.findings))
    assert "ROOFLINE_SINGLE_HOTSPOT" in codes
    report.note(
        "두 번째 Conv 층 하나가 전체 연산의 84% 다. "
        "나머지 층을 아무리 줄여도 전체는 거의 안 줄어든다."
    )


def test_같은_모델도_기계가_바뀌면_병목이_바뀐다(optimized) -> None:
    """이 실습에서 가장 중요한 문장. **병목은 모델의 성질이 아니다.**"""
    from application.optimization.profile_roofline import ProfileRooflineCommand

    profiler = optimized.optimization.profile_roofline()
    slow_memory = ProfileRooflineCommand(
        run_id=optimized.run_id,
        device=DeviceCapability(
            name="대역폭이 좁은 보드",
            peak_gmac_per_second=8.0,
            memory_bandwidth_gb_per_second=0.4,
        ),
    )
    tight_compute = ProfileRooflineCommand(
        run_id=optimized.run_id,
        device=DeviceCapability(
            name="연산기가 약한 보드",
            peak_gmac_per_second=0.2,
            memory_bandwidth_gb_per_second=3.2,
        ),
    )

    memory_view = profiler.execute(slow_memory)
    compute_view = profiler.execute(tight_compute)

    report.block(
        "같은 모델, 다른 보드",
        "\n".join(
            [
                f"  {memory_view.device}",
                f"      → 지배적 병목 {memory_view.dominant_bottleneck}",
                f"  {compute_view.device}",
                f"      → 지배적 병목 {compute_view.dominant_bottleneck}",
            ]
        ),
    )
    assert memory_view.dominant_bottleneck == "MEMORY_BOUND"
    assert compute_view.dominant_bottleneck == "COMPUTE_BOUND"
    report.note(
        "모델은 하나도 안 바뀌었다. 균형점만 바뀌었다. "
        "그러니 '이 모델은 메모리 병목이다'라는 말은 보드를 빼고는 성립하지 않는다."
    )


def test_메모리_병목이_우세하면_양자화가_잘_듣는다() -> None:
    """양자화가 언제 효과가 있는지를 미리 알 수 있다."""
    memory_heavy = RooflineProfile(
        layers=(
            LayerCost(
                name="depthwise",
                kind="Conv1d",
                mac_count=8_000,
                weight_bytes=2_000,
                activation_bytes=120_000,
            ),
            LayerCost(
                name="classifier",
                kind="Linear",
                mac_count=500,
                weight_bytes=2_000,
                activation_bytes=1_000,
            ),
        ),
        device=DeviceCapability(
            name="edge-mcu",
            peak_gmac_per_second=2.0,
            memory_bandwidth_gb_per_second=1.6,
        ),
    )
    findings = RooflinePolicy().inspect(memory_heavy)
    codes = {f.code for f in findings}

    report.block("데이터를 많이 나르는 모델", memory_heavy.render())
    assert memory_heavy.dominant_bottleneck is BottleneckKind.MEMORY_BOUND
    assert "ROOFLINE_MEMORY_BOUND" in codes
    report.note(
        "이런 모델에서는 INT8 이 실제로 빨라진다 — "
        "옮기는 바이트가 1/4 이 되기 때문이다."
    )
    report.note(
        "반대로 우리 모델은 연산 병목이었다. 그래서 INT8 로 크기는 줄었어도 "
        "속도는 오히려 나빠졌다 (실습 4-8). **이 분석이 그 결과를 미리 설명한다.**"
    )


def test_산술_강도는_계산량과_이동량의_비다() -> None:
    layer = LayerCost(
        name="conv",
        kind="Conv1d",
        mac_count=76_800,
        weight_bytes=10_368,
        activation_bytes=5_760,
    )
    assert layer.bytes_moved == 16_128
    assert layer.arithmetic_intensity == pytest.approx(76_800 / 16_128)
    assert layer.bottleneck(machine_balance=1.25) is BottleneckKind.COMPUTE_BOUND
    assert layer.bottleneck(machine_balance=50.0) is BottleneckKind.MEMORY_BOUND
    report.note(
        f"같은 층인데 균형점 1.25 에서는 연산 병목, 50 에서는 메모리 병목이다. "
        f"층의 강도는 {layer.arithmetic_intensity:.2f} 로 고정이고, 판정만 달라진다."
    )
