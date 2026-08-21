"""실습 6-1 — 디바이스에서 발생한 데이터를 Cloud로 보내라.

    pytest -m lesson_6_1 -s

가장 먼저 정할 것은 "어떻게 보낼 것인가"가 아니라 **"무엇을 보낼 것인가"** 다.

전부 보내면 세 군데가 터진다 — 대역폭, 비용, 프라이버시.
그리고 마지막은 되돌릴 수 없다.
"""

from __future__ import annotations

import pytest

from domain.fleet.uplink import UplinkKind, UplinkPolicy
from domain.shared.inspection import Severity
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_1


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_묶어서_올린다(fleet_bare) -> None:
    report.section("실습 6-1 · 디바이스에서 발생한 데이터를 Cloud로 보내라")

    fs.create(fleet_bare)
    view = fs.ingest(fleet_bare, fs.batch())
    report.block("업링크 하나", view.render())

    assert view.accepted
    assert view.uri.startswith("s3://")
    report.note(
        "360건을 한 묶음으로 올렸다. **요청 수가 곧 비용이고, 연결 수립이 곧 전력이다.**"
    )
    report.note(f"저장 위치: {view.uri}")


def test_개인정보는_올라가면_지워지지_않는다() -> None:
    """**이 실습에서 되돌릴 수 없는 유일한 실수다.**"""
    leaky = fs.batch(
        fields=("occurred_at", "predicted_label", "operator_name", "badge_id")
    )
    findings = UplinkPolicy().inspect(leaky)
    report.block("작업자 이름이 섞인 묶음", "\n".join(f"  - {f.describe()}" for f in findings))

    assert "UPLINK_FORBIDDEN_FIELD" in codes(findings)
    assert not UplinkPolicy().accepts(leaky)
    report.note(
        "백업·복제·로그·캐시에 남는다. **올리지 않는 것이 유일한 방법이다.**"
    )
    report.note(
        "그래서 이 검사는 서버가 아니라 **디바이스에서** 먼저 돌아야 한다 — "
        "여기까지 왔다는 건 이미 회선을 건넜다는 뜻이다."
    )


def test_거절된_묶음은_저장되지_않는다(fleet_bare) -> None:
    fs.create(fleet_bare)
    view = fs.ingest(
        fleet_bare,
        fs.batch(fields=("occurred_at", "employee_id")),
    )
    assert not view.accepted
    assert view.uri == ""
    report.note("거절도 결과다. 저장하지 않고, 그 사실을 돌려준다.")


def test_하루_전송_예산을_넘으면_막는다(fleet_bare) -> None:
    """3,000대가 같이 넘으면 회선이 아니라 청구서가 먼저 터진다."""
    fs.create(fleet_bare)
    policy = UplinkPolicy(daily_budget_kib_per_device=200.0)

    first = fs.ingest(fleet_bare, fs.batch(), policy=policy)
    second = fs.ingest(fleet_bare, fs.batch(), policy=policy, part=1)
    third = fs.ingest(fleet_bare, fs.batch(), policy=policy, part=2)

    report.block(
        "같은 디바이스가 세 번 올릴 때",
        "\n".join(
            [
                f"  1회차 누적 {first.sent_today_kib:>7.1f}KiB  {'수락' if first.accepted else '거절'}",
                f"  2회차 누적 {second.sent_today_kib:>7.1f}KiB  {'수락' if second.accepted else '거절'}",
                f"  3회차 누적 {third.sent_today_kib:>7.1f}KiB  {'수락' if third.accepted else '거절'}",
            ]
        ),
    )
    assert first.accepted
    assert not third.accepted
    assert "UPLINK_OVER_DAILY_BUDGET" in {f.code for f in third.findings}
    report.note(
        "누적은 DynamoDB 의 **원자적 증가(ADD)** 로 센다. "
        "읽고-더하고-쓰면 동시에 올라온 두 묶음 중 하나가 사라진다."
    )


def test_너무_잘게_올리면_요청_수가_비용이_된다() -> None:
    findings = UplinkPolicy().inspect(fs.batch(record_count=3, payload_bytes=600))
    assert "UPLINK_TOO_CHATTY" in codes(findings)
    report.note("3건씩 올리면 하루에 수만 번 연결한다. 배터리도 요금도 거기서 나간다.")


def test_체크섬이_없으면_잘린_파일을_못_알아챈다() -> None:
    findings = UplinkPolicy().inspect(fs.batch(checksum=""))
    assert "UPLINK_NO_CHECKSUM" in codes(findings)
    report.note(
        "좁은 회선에서 절반만 올라온 파일도 파일이다. "
        "체크섬 없이는 그것을 온전한 것으로 읽는다."
    )


def test_한_묶음이_너무_크면_통째로_실패한다() -> None:
    findings = UplinkPolicy(max_batch_kib=64.0).inspect(fs.batch())
    assert "UPLINK_BATCH_TOO_LARGE" in codes(findings)
    assert not any(f.severity is Severity.CRITICAL for f in findings)
    report.note(
        "이건 WARNING 이다 — 올려도 되지만 재시도하면 같은 크기를 또 보낸다."
    )


def test_등록되지_않은_디바이스는_올릴_수_없다(fleet_bare) -> None:
    from domain.fleet.errors import DeviceNotFound

    fs.create(fleet_bare)
    with pytest.raises(DeviceNotFound):
        fs.ingest(fleet_bare, fs.batch(device_id="DEV-999"))
    report.note(
        "식별자만 있으면 아무나 올릴 수 있는 구멍을 만들지 않는다. "
        "**등록되지 않은 디바이스의 데이터는 누구 것인지 모른다.**"
    )


def test_올라온_것_자체가_살아_있다는_신호다(fleet_bare) -> None:
    fs.create(fleet_bare)
    fs.ingest(fleet_bare, fs.batch(device_id="DEV-05"))

    from domain.fleet.identifiers import FleetId

    fleet = fleet_bare.fleets.find_by_id(FleetId.of(fs.FLEET_ID))
    assert fleet.device("DEV-05").last_seen_at == "2026-05-23 09:59:59"
    report.note(
        "별도의 헬스체크를 따로 받지 않아도 된다 — "
        "데이터가 올라왔다는 것이 곧 살아 있다는 뜻이다."
    )


def test_올릴_것을_종류로_나눈다() -> None:
    report.block(
        "무엇을 얼마나 올릴 것인가",
        "\n".join(
            [
                f"  {UplinkKind.HEALTH_REPORT.value:<16} 아주 작고 주기적으로",
                f"  {UplinkKind.INFERENCE_LOG.value:<16} 항상 (모듈 5 의 관측)",
                f"  {UplinkKind.RAW_SAMPLE.value:<16} 골라서 (실습 6-4 의 재료)",
                f"  {UplinkKind.INCIDENT.value:<16} 드물게, 원본을 붙여서",
            ]
        ),
    )
    report.note(
        "종류를 나누면 **종류마다 다른 예산과 다른 보관 기간**을 줄 수 있다. "
        "하나로 뭉치면 그게 안 된다."
    )
