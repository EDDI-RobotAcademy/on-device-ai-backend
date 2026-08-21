"""실습 1-11 — 수집 주기와 해상도를 직접 정하라.

    pytest -m lesson_1_11 -s

이 결정은 되돌릴 수 없다.

    빠르게 잰 것은 나중에 느리게 만들 수 있다.
    **느리게 잰 것은 나중에 빠르게 만들 수 없다.**

한 달을 30초 간격으로 모으고 나서 "10초가 필요했다"를 알게 되면
그 한 달은 없는 것이 된다.

그래서 여기서는 여러 주기로 **실제로 다시 뽑아 보고** 무엇이 사라지는지 센다.
그리고 통과하는 것 중 **가장 싼 것**을 고른다 — 가장 빠른 것이 아니라.
"""

from __future__ import annotations

import pytest

from application.data.design_sampling import DesignSamplingCommand
from domain.data.sampling_design import SamplingDesignPolicy, SamplingPlan
from domain.shared.errors import InvariantViolation
from tests.support import quality_scenario as qs
from tests.support import report

pytestmark = pytest.mark.lesson_1_11

INTERVALS = (10.0, 20.0, 30.0, 60.0, 300.0)


@pytest.fixture
def designed(container, model_data):  # noqa: ANN001, ANN201
    qs.prepare_dataset(container, "sampling", model_data.train)
    return container.design_sampling().execute(
        DesignSamplingCommand(
            dataset_id="sampling",
            plans=tuple(
                SamplingPlan(interval_seconds=i, retention_days=30) for i in INTERVALS
            ),
            value_field="active_power_kw",
            policy=SamplingDesignPolicy(target_event_seconds=60.0),
        )
    )


def test_주기마다_무엇이_남는지_직접_센다(designed) -> None:
    report.section("실습 1-11 · 수집 주기와 해상도를 직접 정하라")

    report.block("수집 주기 비교", designed.render())

    assert len(designed.plans) == len(INTERVALS)
    report.note(
        "이론으로 정하지 않았다. **원본을 그 주기로 다시 뽑아 보았다.** "
        "나이퀴스트는 주파수를 말하지만, 모델이 보는 것은 모양이다."
    )


def test_주기를_늘리면_사건이_표본_사이로_빠진다(designed) -> None:
    """이 실습의 본론."""
    fast = designed.plan_at(10.0)
    slow = designed.plan_at(30.0)

    report.block(
        "10초 vs 30초",
        f"  10초 : {fast.row_count:>7,}행  사건 {fast.event_run_count:>3}구간  "
        f"사라진 구간 {fast.lost_event_runs}\n"
        f"  30초 : {slow.row_count:>7,}행  사건 {slow.event_run_count:>3}구간  "
        f"사라진 구간 {slow.lost_event_runs}",
    )

    assert fast.lost_event_runs == 0
    assert slow.lost_event_runs > 0
    report.note(
        f"30초로 모으면 사건 구간 {slow.lost_event_runs}개가 **통째로 사라진다.** "
        "사라진 사건은 라벨도 없고 학습에도 안 들어간다. "
        "그리고 반년 뒤 '왜 이건 못 잡느냐'는 질문으로 돌아온다."
    )


def test_사라진_사건은_데이터를_더_모아도_돌아오지_않는다(designed) -> None:
    slow = designed.plan_at(300.0)
    report.block(
        "5분 간격",
        f"  행 수 {slow.row_count:,} (10초의 1/30)\n"
        f"  사라진 사건 구간 {slow.lost_event_runs}개\n"
        f"  하루 저장량 {slow.bytes_per_day / 1024 / 1024:.2f} MiB",
    )
    assert slow.verdict == "BLOCKED"
    report.note(
        "저장은 30분의 1이다. 그런데 **사건의 대부분이 없다.** "
        "이 파일로 6개월을 모아도 그 6개월에 사건은 없다 — "
        "'데이터가 부족하다'가 아니라 '데이터에 답이 없다'가 된다."
    )


def test_통과하는_것_중_가장_싼_것을_고른다(designed) -> None:
    report.block(
        "선택",
        f"  통과한 설계 {designed.acceptable_count}개\n"
        f"  고른 설계   {designed.cheapest_acceptable}",
    )

    assert designed.acceptable_count >= 1
    assert designed.cheapest_acceptable is not None
    assert designed.cheapest_acceptable.startswith("10초")
    report.note(
        "**가장 빠른 것이 답이 아니다.** 사건을 담을 수 있는 것 중 가장 싼 것이 답이다. "
        "이 데이터에서는 그게 마침 10초였을 뿐이다 — "
        "사건이 더 길었다면 30초가 답이었을 것이다."
    )


def test_무엇을_잡을지_먼저_정해야_주기를_정할_수_있다(container, model_data) -> None:
    """목표 사건 길이를 바꾸면 답이 바뀐다."""
    qs.prepare_dataset(container, "sampling-loose", model_data.train)

    def choose(target: float) -> str | None:
        view = container.design_sampling().execute(
            DesignSamplingCommand(
                dataset_id="sampling-loose",
                plans=(
                    SamplingPlan(interval_seconds=10.0),
                    SamplingPlan(interval_seconds=30.0),
                ),
                policy=SamplingDesignPolicy(target_event_seconds=target),
            )
        )
        return view.cheapest_acceptable

    strict = choose(60.0)
    loose = choose(600.0)

    report.block(
        "목표를 바꾸면 답이 바뀐다",
        f"  '1분짜리는 반드시 잡는다'  → {strict}\n"
        f"  '10분짜리만 잡으면 된다'   → {loose}",
    )

    assert strict is not None
    report.note(
        "**주기를 먼저 논하면 회의가 끝나지 않는다.** "
        "'무엇을 반드시 잡아야 하는가'를 먼저 합의하면 주기는 계산된다. "
        "이 숫자는 코드가 아니라 현장이 정한다."
    )


def test_해상도를_굵게_잡으면_작은_변화가_사라진다(container, model_data) -> None:
    qs.prepare_dataset(container, "sampling-res", model_data.train)
    view = container.design_sampling().execute(
        DesignSamplingCommand(
            dataset_id="sampling-res",
            plans=(
                SamplingPlan(interval_seconds=10.0, value_resolution=0.1),
                SamplingPlan(interval_seconds=10.0, value_resolution=50.0),
            ),
            value_field="active_power_kw",
            policy=SamplingDesignPolicy(target_event_seconds=60.0),
        )
    )

    fine, coarse = view.plans
    report.block(
        "해상도",
        f"  0.1 kW 단위  → 서로 다른 값이 많이 남는다\n"
        f"  50  kW 단위  → 값의 종류가 무너진다\n"
        f"  판정: {fine.verdict} / {coarse.verdict}",
    )

    assert fine.verdict != "BLOCKED"
    assert coarse.verdict == "BLOCKED"
    assert "RESOLUTION_TOO_COARSE" in [f.code for f in view.findings]
    report.note(
        "50kW 단위로 반올림해 저장하면 그보다 작은 변화는 **영원히 사라진다.** "
        "원본을 안 남겨 두면 되살릴 방법이 없다. "
        "이미지에서는 이것이 '가는 균열이 리사이즈에서 뭉개진다'로 나타난다 (실습 3-11)."
    )


def test_주기를_반으로_줄이면_저장은_두_배다() -> None:
    fast = SamplingPlan(interval_seconds=5.0, retention_days=30)
    slow = SamplingPlan(interval_seconds=10.0, retention_days=30)

    report.block(
        "비용",
        f"  5초  : {fast.bytes_retained / 1024 / 1024:>7.1f} MiB / 30일 / 1대\n"
        f"  10초 : {slow.bytes_retained / 1024 / 1024:>7.1f} MiB / 30일 / 1대\n"
        f"  3,000대라면 : {fast.bytes_retained * 3000 / 1024**3:.0f} GiB vs "
        f"{slow.bytes_retained * 3000 / 1024**3:.0f} GiB",
    )

    assert fast.bytes_retained == pytest.approx(slow.bytes_retained * 2)
    report.note(
        "한 대 기준으로 보면 사소하다. **3,000대를 곱하면 사소하지 않다** (모듈 6). "
        "그래서 '일단 빠르게 모으자'는 결정도 공짜가 아니다."
    )


def test_0초_간격은_설계가_아니다() -> None:
    with pytest.raises(InvariantViolation, match="0보다 커야"):
        SamplingPlan(interval_seconds=0.0)
    with pytest.raises(InvariantViolation, match="1일 이상"):
        SamplingPlan(interval_seconds=10.0, retention_days=0)
