"""실습 5-4 — AI가 언제부터 이상해졌는지 찾아라.

    pytest -m lesson_5_4 -s

지금 이상하다는 것은 지금 봐도 안다.
**언제부터인지는 기록이 없으면 영영 모른다.**

그리고 답할 때 두 가지를 구분해야 한다.

    한 번 튄 것(spike)   다음 창에서 돌아온다. 대개 아무 일도 아니다.
    무너진 것(sustained) 연속으로 넘는다. 이쪽이 사건이다.
"""

from __future__ import annotations

import pytest

from domain.operations.health import HealthMetric
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_4


def test_창을_남겨_두었기_때문에_시간선이_있다(deployed) -> None:
    report.section("실습 5-4 · AI가 언제부터 이상해졌는지 찾아라")

    from application.operations.find_onset import GetTimelineQuery

    view = deployed.operations.get_timeline().execute(
        GetTimelineQuery(watch_id=deployed.watch_id)
    )
    report.block("4일치 시간선 (창 8시간)", view.render())

    assert view.window_count == 12
    report.note(
        "이 표를 만들 수 있는지는 지금이 아니라 **배포하던 날** 결정됐다. "
        "관측을 그때 켰기 때문에 1일차 창이 남아 있다."
    )


def test_한_번_튄_것과_무너진_것은_다르다(deployed) -> None:
    """**이 실습의 전부다.**"""
    view = os5.onset(
        deployed.operations,
        metric=HealthMetric.INPUT_PSI,
        threshold=0.2,
        consecutive=3,
    )
    report.block("입력 분포가 언제부터 기준을 넘었는가", view.render())

    assert view.first_exceeded is not None
    assert view.sustained_from is not None
    assert view.first_exceeded != view.sustained_from
    report.note(
        f"처음 넘긴 창은 {view.first_exceeded} 다. "
        "그런데 다음 창에서 돌아왔다 — 도장 부스 문이 열렸던 한 창이다."
    )
    report.note(
        f"실제로 무너진 시점은 {view.sustained_from} 다. "
        "여기서부터 3창 연속으로 기준을 넘었다."
    )
    report.note(
        "이 둘을 구분하지 않으면 알람이 하루에 열 번 울리고, **사람은 곧 알람을 끈다.**"
    )


def test_한_창만_보면_튄_것도_사건이_된다(deployed) -> None:
    spike_only = os5.onset(
        deployed.operations,
        metric=HealthMetric.INPUT_PSI,
        threshold=0.2,
        consecutive=1,
    )
    assert spike_only.sustained_from == spike_only.first_exceeded
    report.note(
        f"연속 조건을 1창으로 두면 무너진 시점이 {spike_only.sustained_from} 로 앞당겨진다. "
        "그건 사건이 아니라 스파이크였다."
    )
    report.note(
        "연속 창 수는 현장이 정한다. 짧게 잡으면 헛알람, 길게 잡으면 늦은 대응이다."
    )


def test_지표마다_무너진_시점이_다르다(deployed) -> None:
    """원인이 다르면 시점도 다르다."""
    drift = os5.onset(
        deployed.operations, metric=HealthMetric.INPUT_PSI, threshold=0.2
    )
    latency = os5.onset(
        deployed.operations, metric=HealthMetric.LATENCY_P95, threshold=0.05
    )

    report.block(
        "두 지표의 무너진 시점",
        "\n".join(
            [
                f"  입력 분포   : {drift.sustained_from}",
                f"  지연시간    : {latency.sustained_from}",
            ]
        ),
    )
    assert drift.sustained_from != latency.sustained_from
    report.note(
        "입력이 먼저 변했고, 지연시간은 하루 뒤에 무너졌다. "
        "**서로 다른 원인이다** — 하나는 여름이고 하나는 팬 고장이다 (실습 5-5, 5-7)."
    )


def test_한_번도_안_넘겼으면_그렇다고_말한다(deployed) -> None:
    calm = os5.onset(
        deployed.operations, metric=HealthMetric.INPUT_PSI, threshold=100.0
    )
    assert calm.first_exceeded is None
    assert not calm.is_sustained
    assert not calm.spike_only
    report.note("기준을 100 으로 두면 아무 일도 없었다. **기준이 판정을 만든다.**")


def test_시간선은_거꾸로_들어오지_않는다(operations_container, deployed) -> None:
    """창이 뒤엉키면 '언제부터'에 답할 수 없다."""
    from domain.shared.errors import InvariantViolation

    windows = os5.windows(operations_container)
    os5.observe(operations_container, windows[3])
    with pytest.raises(InvariantViolation) as caught:
        os5.observe(operations_container, windows[1])
    report.note(str(caught.value))
