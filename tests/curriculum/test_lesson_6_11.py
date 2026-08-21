"""실습 6-11 — 수천 대의 AI를 동시에 운영할 수 있는 구조를 만들어라.

    pytest -m lesson_6_11 -s

규모가 커지면 질문 자체가 바뀐다.

    한 대   "이 모델이 잘 돌고 있는가"
    수천 대 "**몇 대가** 어느 버전인가 / 연락이 안 되는가 / 실패하는가"

그리고 두 가지가 늘 참이다.

    1. **몇 대는 언제나 연락이 안 된다.**
    2. **버전은 절대 한 줄로 정렬되지 않는다.**
"""

from __future__ import annotations

import pytest

from domain.fleet.device import DeviceStatus, FleetHealthPolicy, FleetSummary
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_11


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def summary(**overrides) -> FleetSummary:  # noqa: ANN003
    base: dict[str, object] = dict(
        total=3_000,
        by_status={"HEALTHY": 2_900, "STALE": 80, "UNREACHABLE": 20},
        by_version={"v2.0.0": 2_700, "v1.9.0": 300},
        by_group={"line-a": 1_500, "line-b": 1_500},
        never_reported=0,
    )
    base.update(overrides)
    return FleetSummary(**base)  # type: ignore[arg-type]


def test_0을_요구하는_정책은_매일_실패한다() -> None:
    """**몇 대는 언제나 연락이 안 된다.**"""
    strict = FleetHealthPolicy(max_stale_ratio=0.0).inspect(summary())
    realistic = FleetHealthPolicy(max_stale_ratio=0.05).inspect(summary())

    report.section("실습 6-11 · 수천 대의 AI를 동시에 운영할 수 있는 구조")
    report.block(
        "3,000대 중 100대가 연락 안 됨 (3.3%)",
        "\n".join(
            [
                f"  기준 0%  → {sorted(codes(strict))}",
                f"  기준 5%  → {sorted(codes(realistic))}",
            ]
        ),
    )
    assert "FLEET_TOO_MANY_STALE" in codes(strict)
    assert not codes(realistic)
    report.note(
        "꺼져 있고, 정비 중이고, 회선이 끊겼다. **그건 사고가 아니라 상태다.**"
    )
    report.note(
        "0을 요구하면 알람이 매일 울리고, 사람은 알람을 끈다 — 모듈 5-4 와 같은 이야기다."
    )


def test_버전이_너무_여러_종이면_원인을_못_가린다() -> None:
    skewed = FleetHealthPolicy().inspect(
        summary(
            by_version={
                "v2.0.0": 900,
                "v1.9.0": 800,
                "v1.8.0": 700,
                "v1.7.0": 600,
            }
        )
    )
    report.block("네 버전이 동시에", "\n".join(f"  - {f.describe()}" for f in skewed))

    assert "FLEET_VERSION_SKEW" in codes(skewed)
    report.note(
        "문제가 생겨도 **어느 버전의 문제인지 가릴 수 없다.** "
        "그리고 모듈 5 의 로그는 네 모델이 섞여 들어온다."
    )


def test_배포가_중간에_멈춰_있는_것도_상태다() -> None:
    stalled = FleetHealthPolicy().inspect(
        summary(by_version={"v2.0.0": 1_500, "v1.9.0": 1_500})
    )
    assert "FLEET_ROLLOUT_STALLED" in codes(stalled)
    report.note(
        "절반씩 갈려 있다. **끝내거나 되돌려야 한다** — "
        "이 상태로 두는 것이 가장 나쁘다."
    )


def test_설치가_안_끝난_디바이스를_찾는다() -> None:
    findings = FleetHealthPolicy().inspect(summary(never_reported=42))
    assert "FLEET_NEVER_REPORTED" in codes(findings)
    report.note(
        "등록은 됐는데 한 번도 안 올라왔다. "
        "대개 설치가 안 끝났거나 자격증명이 안 들어간 것이다 — "
        "**배포 대상 수를 세는 순간 이 42대가 실패로 잡힌다.**"
    )


def test_오래_연락_없는_디바이스를_표시한다(fleet_env) -> None:
    view = fs.sweep(fleet_env.fleet)
    report.block("이틀 뒤 훑기", view.render())

    assert view.stale_ratio > 0
    report.note(
        "마지막 보고 시각만 보고 표시한다. **Domain 은 시계를 모른다** — "
        "기준 시각을 받아서 비교만 한다."
    )
    report.note(
        "이렇게 표시된 디바이스는 배포 대상에서 빠지고(6-8), "
        "학습 데이터에서도 빠진다(6-4)."
    )


def test_격리된_디바이스는_훑기가_건드리지_않는다(fleet_env) -> None:
    fs.mark(fleet_env.fleet, "DEV-03", DeviceStatus.QUARANTINED, "모듈 5 격리")
    fs.sweep(fleet_env.fleet)

    from domain.fleet.identifiers import FleetId

    fleet = fleet_env.fleet.fleets.find_by_id(FleetId.of(fs.FLEET_ID))
    assert fleet.device("DEV-03").status is DeviceStatus.QUARANTINED
    report.note(
        "격리는 **사람이 정한 상태다.** 자동 훑기가 그것을 덮어쓰면 "
        "멈춰 세운 디바이스가 슬그머니 배포 대상으로 돌아온다."
    )


def test_3000대도_같은_화면에_들어간다() -> None:
    """**목록이 아니라 집계이기 때문이다.**"""
    small = summary(total=24, by_status={"HEALTHY": 24}, by_version={"v2.0.0": 24},
                    by_group={"line-a": 24})
    large = summary()

    report.block("24대", small.render())
    report.block("3,000대", large.render())

    assert len(small.render().splitlines()) < 15
    assert len(large.render().splitlines()) < 15
    report.note("줄 수가 대수에 비례하지 않는다. **그게 수천 대를 다루는 유일한 방법이다.**")


def test_부분_실패가_정상이다(fleet_env) -> None:
    """이 모듈 전체의 전제."""
    fs.plan(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.advance(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.advance(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.apply_outcomes(fleet_env.fleet)

    from application.fleet.release_and_rollout import GetRolloutQuery

    rollout = fleet_env.fleet.get_rollout().execute(GetRolloutQuery(rollout_id="ro-1"))
    view = fs.summarize(fleet_env.fleet)

    report.block("배포를 다 돌린 뒤", view.render())
    report.note(
        f"도달 {rollout.coverage:.1%}. 나머지는 꺼져 있거나 아직 답이 없다."
    )
    assert rollout.coverage < 1.0
    assert view.version_count >= 2
    report.note(
        "**version skew 를 0으로 만들려는 설계는 실패한다.** "
        "할 수 있는 것은 얼마나 벌어져 있는지 아는 것뿐이다."
    )
    report.note(
        "그리고 그 숫자가 다음 롤아웃의 입력이 된다 — "
        "안 받은 디바이스만 다시 대상으로 잡으면 된다."
    )


def test_Aggregate_하나가_3000대를_드는_것의_한계(fleet_env) -> None:
    """**정직하게 적어 둔다.**"""
    report.block(
        "지금 설계",
        "\n".join(
            [
                "  Fleet Aggregate 가 디바이스 목록을 통째로 들고 있다",
                "  배포 판정이 전체 상태를 봐야 하기 때문이다 —",
                "  '몇 대가 어느 버전인가' 를 모르면 다음 wave 를 정할 수 없다",
            ]
        ),
    )
    report.note(
        "수만 대가 되면 이 설계는 무너진다. 한 Aggregate 를 읽고 쓰는 데 "
        "수만 건이 오간다."
    )
    report.note(
        "그때는 Fleet 을 **사이트별로 쪼개고**, 집계를 **읽기 전용 모델**로 뺀다. "
        "DeviceRegistry Port 가 이미 그 자리를 잡아 두고 있다 — "
        "지금도 조회는 그쪽이 담당한다."
    )
    assert fleet_env.fleet.registry is not None
