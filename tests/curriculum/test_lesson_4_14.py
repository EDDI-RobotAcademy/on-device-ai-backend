"""실습 4-14 — 한 장씩 볼 때와 묶어서 볼 때는 다르다.

    pytest -m lesson_4_14 -s

실습 4-1 에서 "배치 32로 재면 안 된다"고 했다. 여기서 **왜인지를 숫자로 본다.**

배치를 키우면 두 숫자가 반대로 움직인다.

    표본당 시간      짧아진다   ← 처리량(throughput)
    첫 답까지 시간   길어진다   ← 지연시간(latency)

배치는 다 모여야 계산이 시작된다.
그래서 배치 64로 돌리면, 첫 번째 표본은 나머지 63개를 기다린다.

현장의 사이클 타임을 지키는 것은 **뒤의 숫자**다.
그리고 이 둘을 헷갈리면 "표본당 0.001ms 나옵니다"라고 보고하고
현장에서 사이클을 놓친다.
"""

from __future__ import annotations

import pytest

from application.optimization.measure_resources import ScaleBatchCommand
from domain.optimization.resource import BatchPoint, BatchScaling
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_4_14


@pytest.fixture(scope="module")
def scaling(optimized):  # noqa: ANN001, ANN201
    return optimized.optimization.scale_batch().execute(
        ScaleBatchCommand(
            run_id=optimized.run_id,
            label="TFLITE/FP32",
            batch_sizes=(1, 4, 16, 64),
            cycle_time_ms=0.01,
        )
    )


def test_배치를_키우면_두_숫자가_반대로_움직인다(scaling) -> None:
    report.section("실습 4-14 · 한 장씩 볼 때와 묶어서 볼 때는 다르다")

    report.block("배치 크기별", scaling.render())

    single_per_sample, single_first = scaling.at(1)
    large_per_sample, large_first = scaling.at(64)

    assert large_per_sample < single_per_sample
    assert large_first > single_first
    report.note(
        f"표본당 시간은 {single_per_sample:.4f}ms → {large_per_sample:.4f}ms 로 짧아졌다. "
        f"**첫 답까지는 {single_first:.4f}ms → {large_first:.4f}ms 로 길어졌다.** "
        "같은 측정에서 나온 두 숫자가 반대 방향이다."
    )


def test_처리량을_지연시간이라고_부르면_사이클을_놓친다(scaling) -> None:
    """이 실습의 본론."""
    report.block(
        "무엇을 지켜야 하는가",
        f"  처리량 이득     {scaling.throughput_gain:.1f}배\n"
        f"  첫 답까지 대가  {scaling.latency_cost:.1f}배\n"
        f"  사이클 타임     {scaling.cycle_time_ms:g}ms",
    )

    assert scaling.throughput_gain > 1.0
    assert scaling.latency_cost > scaling.throughput_gain
    report.note(
        "처리량은 3.6배가 되었는데 첫 답까지는 17배가 되었다. "
        "**이득보다 대가가 크다.** "
        "라인이 표본을 한 장씩 보내는 구조라면 배치는 손해다."
    )


def test_사이클_타임을_넘기면_막는다(scaling) -> None:
    codes = [f.code for f in scaling.findings]
    report.block(
        "소견", "\n".join(f"  {f.describe()}" for f in scaling.findings)
    )

    assert "BATCH_MISSES_CYCLE_TIME" in codes
    report.note(
        "표본당 시간으로 보면 여유가 넘친다. "
        "**첫 답까지로 보면 사이클을 놓친다.** "
        "판정은 뒤의 숫자로 한다."
    )


def test_배치가_의미있는_구조도_있다(scaling) -> None:
    assert "BATCH_THROUGHPUT_GAIN" in [f.code for f in scaling.findings]
    report.note(
        "**배치가 항상 나쁜 것은 아니다.** "
        "여러 대의 디바이스가 클라우드 한 곳으로 보내고 거기서 묶어 처리한다면 "
        "처리량이 곧 비용이다 (모듈 6). "
        "나쁜 것은 배치가 아니라 **배치로 잰 숫자를 디바이스 지연시간이라고 부르는 것**이다."
    )


def test_기준은_언제나_표본_하나다() -> None:
    with pytest.raises(InvariantViolation, match="배치 1이어야"):
        BatchScaling(
            points=(
                BatchPoint(batch_size=4, p50_ms=1.0, p95_ms=1.2),
                BatchPoint(batch_size=16, p50_ms=3.0, p95_ms=3.5),
            )
        )
    report.note(
        "배치 4와 16만 재면 '4가 더 빠르다'까지만 알 수 있다. "
        "**한 장씩 넣었을 때가 없으면 현장 숫자가 없는 것이다.**"
    )


def test_비교는_최소_두_지점이_필요하다() -> None:
    with pytest.raises(InvariantViolation, match="최소 두 지점"):
        BatchScaling(points=(BatchPoint(batch_size=1, p50_ms=1.0, p95_ms=1.2),))
