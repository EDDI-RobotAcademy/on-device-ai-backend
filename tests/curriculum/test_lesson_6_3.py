"""실습 6-3 — 수천 개의 디바이스 데이터를 하나로 관리하라.

    pytest -m lesson_6_3 -s

디바이스가 3,000대면 **목록은 답이 아니다.** 아무도 3,000줄을 읽지 않는다.
필요한 것은 집계다.
"""

from __future__ import annotations

import pytest

from domain.fleet.device import Device, DeviceStatus
from domain.fleet.identifiers import DeviceId, FleetId
from domain.shared.errors import InvariantViolation
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_3


def test_목록이_아니라_집계다(fleet_env) -> None:
    report.section("실습 6-3 · 수천 개의 디바이스 데이터를 하나로 관리하라")

    view = fs.summarize(fleet_env.fleet)
    report.block("아침에 보는 화면", view.render())

    assert view.size == fs.DEVICE_COUNT
    report.note(
        "24대든 3,000대든 **화면 크기가 같다.** 목록이면 그렇지 않다."
    )


def test_한_플릿_조회는_Query_한_번이다(fleet_env) -> None:
    """키 설계가 여기서도 문제다. S3 와 같은 이유다."""
    registry = fleet_env.fleet.registry
    devices = registry.list_devices(FleetId.of(fs.FLEET_ID))

    report.block(
        "DynamoDB 키 설계",
        "\n".join(
            [
                "  PK = fleet_id    한 플릿의 디바이스를 한 번에 긁는다",
                "  SK = device_id   한 대를 바로 찾는다",
                "",
                f"  → 이 플릿의 {len(devices)}대를 **Query 한 번**으로 가져왔다",
            ]
        ),
    )
    assert len(devices) == fs.DEVICE_COUNT
    report.note(
        "PK 를 device_id 로 두면 같은 질문이 **Scan** 이 된다 — "
        "1,000대면 1,000개를 다 읽고 필터링한다."
    )


def test_한_대를_바로_찾는다(fleet_env) -> None:
    registry = fleet_env.fleet.registry
    found = registry.find(FleetId.of(fs.FLEET_ID), DeviceId.of("DEV-07"))

    assert found is not None
    assert found.group == "line-a"
    assert registry.find(FleetId.of(fs.FLEET_ID), DeviceId.of("DEV-999")) is None
    report.note("없는 디바이스는 None 이다. 예외가 아니다 — 조회에서는 흔한 일이다.")


def test_같은_식별자를_두_대가_쓰면_막는다(fleet_bare) -> None:
    from domain.fleet.fleet import Fleet

    fleet = Fleet.create(FleetId.of("dup"), "중복 시험")
    fleet.register(Device(device_id="DEV-00", group="g1"))
    with pytest.raises(InvariantViolation) as caught:
        fleet.register(Device(device_id="DEV-00", group="g2"))
    report.note(str(caught.value))
    report.note(
        "현장에서 실제로 일어난다 — 이미지를 복제해 굽거나, 교체 장비에 옛 ID 를 넣는다."
    )


def test_그룹이_있어야_단계적_배포를_할_수_있다(fleet_env) -> None:
    from domain.fleet.identifiers import FleetId as FID

    fleet = fleet_env.fleet.fleets.find_by_id(FID.of(fs.FLEET_ID))
    report.block(
        "그룹",
        "\n".join(
            f"  {group:<10}{len(fleet.devices_in(group)):>4}대" for group in fs.GROUPS
        ),
    )
    assert len(fleet.devices_in("pilot")) == 2
    report.note(
        "그룹 없이 나누면 무작위가 된다. "
        "**파일럿은 '아무 2대'가 아니라 '봐 줄 사람이 있는 2대'여야 한다.**"
    )


def test_격리된_디바이스는_배포_대상에서_빠진다(fleet_env) -> None:
    from domain.fleet.identifiers import FleetId as FID

    fs.mark(fleet_env.fleet, "DEV-03", DeviceStatus.QUARANTINED, "모듈 5 격리")
    fleet = fleet_env.fleet.fleets.find_by_id(FID.of(fs.FLEET_ID))

    reachable = {d.device_id for d in fleet.reachable_devices()}
    assert "DEV-03" not in reachable
    report.note(
        "모듈 5 가 판단을 멈춰 세운 디바이스다. "
        "새 모델을 보내도 안 받고, **실패로 세면 실패율이 거짓말을 한다.**"
    )


def test_보고가_거꾸로_들어오면_막는다(fleet_env) -> None:
    from domain.fleet.identifiers import FleetId as FID

    fleet = fleet_env.fleet.fleets.find_by_id(FID.of(fs.FLEET_ID))
    fleet.report("DEV-01", seen_at="2026-05-24 10:00:00")

    with pytest.raises(InvariantViolation) as caught:
        fleet.report("DEV-01", seen_at="2026-05-23 10:00:00")
    report.note(str(caught.value))
    report.note(
        "디바이스 시계는 실제로 어긋난다. 그걸 그대로 받으면 "
        "'마지막으로 본 시각'이 과거로 되돌아간다."
    )


def test_버전은_디바이스가_말하는_것을_믿는다(fleet_env) -> None:
    """**서버가 '보냈으니 올라갔겠지'라고 적지 않는다.**"""
    from domain.fleet.identifiers import FleetId as FID

    fleet = fleet_env.fleet.fleets.find_by_id(FID.of(fs.FLEET_ID))
    before = fleet.device("DEV-01").current_version
    fleet.report("DEV-01", seen_at="2026-05-24 10:00:00", version="v2.0.0")
    after = fleet.device("DEV-01").current_version

    report.block("버전 보고", f"  {before} → {after}")
    assert after == "v2.0.0"
    events = [e.event_name for e in fleet.pull_events()]
    assert "DeviceVersionChanged" in events
    report.note(
        "서버가 적어 두면 **실제로 안 올라간 대수를 영영 모른다.** "
        "그게 6-8 의 실패율을 거짓말로 만든다."
    )
