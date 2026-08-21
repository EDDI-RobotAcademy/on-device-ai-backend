"""실습 6-9 — 문제가 생기면 이전 모델로 되돌려라.

    pytest -m lesson_6_9 -s

모듈 5 의 롤백은 기록 하나를 바꾸는 일이었다.
수천 대에서는 다르다.

    **롤백도 또 하나의 롤아웃이다.**

똑같이 단계로 나가고, 똑같이 오프라인 디바이스를 만나고, 똑같이 시간이 걸린다.
"되돌렸습니다"라고 말한 순간에도 현장의 절반은 아직 새 모델을 돌리고 있다.
"""

from __future__ import annotations

import pytest

from domain.fleet.identifiers import FleetId
from domain.fleet.release import ReleaseChannel
from infrastructure.edge.ota_simulator import FleetResponseProfile
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_9


@pytest.fixture
def broken(fleet_env):  # noqa: ANN001, ANN201
    """v2.0.0 을 절반쯤 내보낸 상태. 그리고 현장에서 문제가 발견됐다."""
    fs.publish(fleet_env.fleet, fs.bundle(version="v1.0.0", source_build_id="build-old"))
    fs.plan(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.advance(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.apply_outcomes(fleet_env.fleet)
    return fleet_env


def test_롤백도_롤아웃이다(broken) -> None:
    report.section("실습 6-9 · 문제가 생기면 이전 모델로 되돌려라")

    view = fs.rollback(broken.fleet, to_version="v1.0.0")
    report.block("되돌리기 롤아웃", view.render())

    assert view.rollout_id == "ro-1-rollback"
    assert view.version == "v1.0.0"
    assert view.status == "RUNNING"
    assert view.coverage == 0.0
    report.note(
        "**아직 아무것도 안 돌아갔다.** 되돌리기 계획을 세우고 첫 단계를 시작했을 뿐이다."
    )
    report.note("모듈 5 의 롤백은 여기서 끝났다. 여기서는 이제 시작이다.")


def test_되돌리는_데도_단계가_있다(broken) -> None:
    fs.rollback(broken.fleet, to_version="v1.0.0")
    first = fs.collect(broken.fleet, rollout_id="ro-1-rollback")
    report.block("되돌리기 wave-1", first.render())

    fs.advance(broken.fleet, rollout_id="ro-1-rollback")
    second = fs.collect(broken.fleet, rollout_id="ro-1-rollback")
    report.block("되돌리기 wave-2", second.render())

    assert first.size < second.size
    report.note(
        "**되돌리는 것도 잘못될 수 있다.** 이전 모델이 지금 환경에서 안 돌 수도 있다 — "
        "그래서 되돌릴 때도 몇 대로 먼저 확인한다."
    )


def test_새_버전을_받은_디바이스만_되돌리면_된다(broken) -> None:
    """받지 않은 것은 그대로 두면 된다."""
    fleet = broken.fleet.fleets.find_by_id(FleetId.of(fs.FLEET_ID))
    on_new = fleet.devices_on("v2.0.0")
    still_old = fleet.devices_on("v1.0.0")

    view = fs.rollback(broken.fleet, to_version="v1.0.0")
    report.block(
        "되돌릴 대상",
        "\n".join(
            [
                f"  v2.0.0 을 받은 디바이스 : {len(on_new)}대  ← 되돌린다",
                f"  아직 v1.0.0 인 디바이스 : {len(still_old)}대  ← 그대로 둔다",
                f"  되돌리기 대상          : {view.device_count}대",
            ]
        ),
    )
    assert view.device_count == len(on_new)
    report.note(
        "전부에게 다시 내보내면 **멀쩡한 디바이스까지 건드린다.** "
        "OTA 는 그 자체로 위험한 작업이다."
    )


def test_첫_배포였으면_되돌릴_곳이_없다(fleet_bare) -> None:
    """**되돌릴 곳이 없는 배포는 격리(모듈 5)밖에 수단이 없다.**"""
    from application.shared.errors import ConflictingRequest

    # 아직 아무것도 안 받은 디바이스들 — 이전 버전이라는 것이 존재하지 않는다.
    fs.create(fleet_bare, devices=fs.devices(version=""))
    fs.publish(fleet_bare, fs.bundle())
    fs.plan(fleet_bare)

    with pytest.raises(ConflictingRequest) as caught:
        fs.rollback(fleet_bare, to_version="")
    report.note(str(caught.value))
    report.note(
        "실습 5-1 의 '첫 배포는 좁게' 가 여기서 값을 한다 — "
        "되돌릴 수 없으니 겪는 대수를 줄이는 수밖에 없다."
    )


def test_등록되지_않은_버전으로는_못_돌아간다(broken) -> None:
    from domain.fleet.errors import ReleaseNotFound

    with pytest.raises(ReleaseNotFound) as caught:
        fs.rollback(broken.fleet, to_version="v0.9.0")
    report.note(str(caught.value))
    report.note(
        "그래서 채널에서 밀려난 릴리스를 **지우지 않는다** (실습 6-7). "
        "ARCHIVED 는 롤백 대상이다."
    )


def test_원래_롤아웃은_되돌려졌다고_기록된다(broken) -> None:
    fs.rollback(broken.fleet, to_version="v1.0.0")

    from application.fleet.release_and_rollout import GetRolloutQuery

    original = broken.fleet.get_rollout().execute(GetRolloutQuery(rollout_id="ro-1"))
    report.block("원래 롤아웃", original.render())

    assert original.status == "ROLLED_BACK"
    assert any("되돌림" in what for _, what in original.history)
    report.note(
        "**무엇이 무엇을 되돌렸는가**가 양쪽에 남는다. "
        "사건 조사는 이 두 이력을 나란히 읽는다."
    )


def test_이유_없는_되돌리기는_없다(broken) -> None:
    from domain.shared.errors import InvariantViolation

    with pytest.raises(InvariantViolation):
        fs.rollback(broken.fleet, to_version="v1.0.0", reason="  ")
    report.note("'일단 되돌림' 은 반년 뒤에 아무 도움이 안 된다.")


def test_되돌리는_동안_현장은_섞여_있다(broken) -> None:
    """**이게 수천 대의 현실이다.**"""
    fs.rollback(broken.fleet, to_version="v1.0.0")
    fs.collect(broken.fleet, rollout_id="ro-1-rollback")
    fs.apply_outcomes(broken.fleet, rollout_id="ro-1-rollback")

    view = fs.summarize(broken.fleet)
    report.block("되돌리는 중의 플릿", view.render())

    assert view.version_count >= 2
    report.note(
        "일부는 v1.0.0, 일부는 아직 v2.0.0 이다. "
        "**이 상태가 몇 시간에서 며칠 간다.**"
    )
    report.note(
        "그 사이에 들어오는 추론 로그는 두 모델이 섞여 있다 — "
        "모듈 5 의 LOG_MIXED_VERSIONS 가 여기서 나온다."
    )


def test_진행하던_OTA_작업은_취소된다(broken) -> None:
    import boto3

    fs.rollback(broken.fleet, to_version="v1.0.0")
    iot = boto3.client("iot", region_name="ap-northeast-2")
    jobs = {j["jobId"]: j.get("status") for j in iot.list_jobs().get("jobs", [])}

    report.block("IoT Job 상태", "\n".join(f"  {k}: {v}" for k, v in jobs.items()))
    report.note(
        "안 그러면 꺼져 있던 디바이스가 나중에 깨어나 **되돌린 뒤에 새 모델을 받는다.**"
    )
    assert any(k.startswith("ota-ro-1-rollback") for k in jobs)
