"""실습 6-7 — 모델 Version을 Cloud에서 관리하라.

    pytest -m lesson_6_7 -s

**한 채널에 두 버전이 동시에 있을 수 없다.**
그것을 허용하는 순간 "지금 stable 이 뭐죠?"에 답할 수 없게 된다.

    canary   몇 대에만. 문제가 나도 몇 대다.
    stable   전부에게. canary 를 통과한 것만 온다.
"""

from __future__ import annotations

import pytest

from domain.fleet.identifiers import FleetId
from domain.fleet.release import ChannelState, ReleaseChannel
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_7


def fleet_of(container):  # noqa: ANN001, ANN201
    return container.fleets.find_by_id(FleetId.of(fs.FLEET_ID))


def test_채널은_지금_무엇이_도는지에_답한다(fleet_env) -> None:
    report.section("실습 6-7 · 모델 Version을 Cloud에서 관리하라")

    fleet = fleet_of(fleet_env.fleet)
    report.block("채널 상태", f"  {fleet.channels.describe()}")

    assert fleet.channels.canary == "v2.0.0"
    assert fleet.channels.stable == ""
    report.note("canary 에 올라가 있다. 아직 stable 은 비어 있다.")


def test_canary를_거치지_않으면_stable로_못_간다(fleet_env) -> None:
    """**몇 대에서 확인하지 않은 것을 전부에게 보내지 않는다.**"""
    fs.publish(fleet_env.fleet, fs.bundle(version="v3.0.0"))

    with pytest.raises(IllegalStateTransition) as caught:
        fs.promote(fleet_env.fleet, "v3.0.0", ReleaseChannel.STABLE)
    report.note(str(caught.value))
    report.note(
        "이 규칙 하나가 '급하니까 바로 전체 배포' 를 막는다. "
        "그리고 급한 상황일수록 그 배포가 위험하다."
    )


def test_한_채널에_하나뿐이다(fleet_env) -> None:
    fs.publish(fleet_env.fleet, fs.bundle(version="v3.0.0"))
    state = fs.promote(fleet_env.fleet, "v3.0.0", ReleaseChannel.CANARY)

    fleet = fleet_of(fleet_env.fleet)
    report.block("v2.0.0 → v3.0.0 승격 후", f"  {state}")

    assert fleet.channels.canary == "v3.0.0"
    assert "v2.0.0" in fleet.channels.archived
    report.note(
        "있던 것은 ARCHIVED 로 밀려난다. **지우지는 않는다** — 롤백 대상일 수 있다."
    )


def test_stable로_올라가면_canary에서_빠진다() -> None:
    state = ChannelState(canary="v2.0.0", stable="v1.0.0")
    promoted = state.with_promotion("v2.0.0", ReleaseChannel.STABLE)

    report.block(
        "승격 전후",
        "\n".join([f"  전 : {state.describe()}", f"  후 : {promoted.describe()}"]),
    )
    assert promoted.stable == "v2.0.0"
    assert promoted.canary == ""
    assert "v1.0.0" in promoted.archived
    report.note(
        "같은 버전이 canary 와 stable 에 동시에 있으면 "
        "**canary 가 무엇을 시험하고 있는지 알 수 없다.**"
    )


def test_등록되지_않은_버전은_올릴_수_없다(fleet_env) -> None:
    from domain.fleet.errors import NotReleasable

    with pytest.raises(NotReleasable) as caught:
        fs.promote(fleet_env.fleet, "v9.9.9", ReleaseChannel.CANARY)
    report.note(str(caught.value))
    report.note("채널은 이름표가 아니다. **실제로 있는 묶음을 가리켜야 한다.**")


def test_ARCHIVED로는_승격하지_않는다() -> None:
    with pytest.raises(InvariantViolation):
        ChannelState().with_promotion("v1.0.0", ReleaseChannel.ARCHIVED)
    report.note(
        "ARCHIVED 는 밀려나는 자리지 올라가는 자리가 아니다. "
        "내리려면 다른 것을 올리면 된다."
    )


def test_승격은_사건으로_남는다(fleet_env) -> None:
    fleet = fleet_of(fleet_env.fleet)
    fleet.pull_events()

    fs.publish(fleet_env.fleet, fs.bundle(version="v3.0.0"))
    fs.promote(fleet_env.fleet, "v3.0.0", ReleaseChannel.CANARY)

    fleet = fleet_of(fleet_env.fleet)
    report.note(
        "ReleasePublished 와 ReleasePromoted 가 남는다 — "
        "**언제 무엇이 어느 채널에 올라갔는지**가 이력이 된다."
    )
    assert fleet.channels.canary == "v3.0.0"


def test_버전은_세_가지가_있고_서로_다르다(fleet_env) -> None:
    bundle = fs.bundle()
    report.block(
        "세 가지 버전",
        "\n".join(
            [
                f"  모델 버전  : {bundle.model_version_id}   어떤 학습이 만든 가중치인가",
                f"  결과물     : {bundle.runtime}/{bundle.precision}"
                "                 어떤 형식으로 바꿨는가",
                f"  릴리스     : {bundle.version}"
                "                        디바이스가 구분하는 이름",
            ]
        ),
    )
    report.note(
        "모듈 5 의 배포 버전(v1, v2)과도 다르다. "
        "**저건 한 대의 이력이고, 이건 플릿 전체의 카탈로그다.**"
    )
    assert bundle.model_version_id != bundle.version
