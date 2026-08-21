"""실습 6-8 — OTA로 현장의 AI를 업데이트하라.

    pytest -m lesson_6_8 -s

모듈 5 의 배포와 결정적으로 다른 점.

    **한 번에 다 안 간다.** 몇 대는 꺼져 있고, 몇 대는 회선이 끊겼다.
    "배포 완료"는 100% 가 아니라 **"받을 수 있는 것은 다 받았다"** 이다.

그래서 단계(wave)로 나눈다. 그리고 각 단계 뒤에 멈출 수 있어야 의미가 있다.
"""

from __future__ import annotations

import pytest

from domain.fleet.rollout import DeviceOutcome, Rollout, RolloutPolicy, Wave, WaveResult
from domain.fleet.identifiers import RolloutId
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from infrastructure.edge.ota_simulator import FleetResponseProfile
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_8


def wave(name: str, count: int, start: int = 0) -> Wave:
    return Wave(
        name=name, device_ids=tuple(f"DEV-{n:02d}" for n in range(start, start + count))
    )


def result(**outcomes: DeviceOutcome) -> WaveResult:
    target = wave("w", len(outcomes))
    mapping = dict(zip(target.device_ids, outcomes.values(), strict=True))
    return WaveResult(wave=target, outcomes=mapping)


def test_단계로_나눠_내보낸다(fleet_env) -> None:
    report.section("실습 6-8 · OTA로 현장의 AI를 업데이트하라")

    view = fs.plan(fleet_env.fleet)
    report.block("롤아웃 계획", view.render())

    assert view.device_count == fs.DEVICE_COUNT
    report.note(
        "pilot 2대 → line-a 8대 → line-b 14대. "
        "**그룹 순서가 곧 단계 순서다** (실습 6-3)."
    )
    report.note("첫 단계에서 문제가 나면 2대만 겪는다.")


def test_100퍼센트는_오지_않는다(fleet_env) -> None:
    """**이 실습의 전제다.**"""
    fs.plan(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.advance(fleet_env.fleet)
    fs.collect(fleet_env.fleet)
    fs.advance(fleet_env.fleet)
    view = fs.collect(fleet_env.fleet)

    from application.fleet.release_and_rollout import GetRolloutQuery

    final = fleet_env.fleet.get_rollout().execute(GetRolloutQuery(rollout_id="ro-1"))
    report.block("3단계 모두 끝난 뒤", final.render())

    assert final.coverage < 1.0
    assert final.unreachable > 0
    report.note(
        f"도달 {final.coverage:.1%}. 나머지는 꺼져 있거나 아직 답이 없다."
    )
    report.note(
        "**100% 를 기다리는 배포는 영원히 안 끝난다.** "
        "받을 수 있는 것을 다 받았으면 끝난 것이다."
    )


def test_실패율의_분모는_전체가_아니라_시도한_것이다() -> None:
    """**꺼져 있는 디바이스를 실패로 세면 아무 배포도 통과하지 못한다.**"""
    mixed = result(
        a=DeviceOutcome.SUCCEEDED,
        b=DeviceOutcome.SUCCEEDED,
        c=DeviceOutcome.FAILED,
        d=DeviceOutcome.UNREACHABLE,
        e=DeviceOutcome.UNREACHABLE,
        f=DeviceOutcome.PENDING,
    )
    report.block(
        "6대 중 성공 2 · 실패 1 · 미도달 2 · 대기 1",
        "\n".join(
            [
                f"  전체 기준 실패율 : {1 / 6:.1%}",
                f"  시도 기준 실패율 : {mixed.failure_ratio:.1%}   ← 이쪽을 쓴다",
            ]
        ),
    )
    assert mixed.attempted == 3
    assert mixed.failure_ratio == pytest.approx(1 / 3)
    report.note(
        "미도달과 대기는 **시도가 아니다.** 모델 문제가 아니라 회선·전원 문제다."
    )


def test_실패가_기준을_넘으면_스스로_멈춘다(fleet_env) -> None:
    """**사람이 대시보드를 보고 결정하기까지 기다리지 않는다.**"""
    fleet_env.fleet.ota.set_profile(
        FleetResponseProfile(failure_rate=0.5, offline_rate=0.0, pending_rate=0.0)
    )
    fs.plan(fleet_env.fleet)
    view = fs.collect(fleet_env.fleet)

    report.block("절반이 실패한 wave-1", view.render())
    assert view.halted
    assert "OTA_FAILURE_RATE" in {f.code for f in view.findings}
    report.note("그 사이에도 나머지 22대로는 안 나간다. 그게 단계로 나눈 이유다.")


def test_멈춘_뒤에는_다음_단계로_못_간다(fleet_env) -> None:
    from domain.fleet.errors import RolloutHalted

    fleet_env.fleet.ota.set_profile(
        FleetResponseProfile(failure_rate=0.5, offline_rate=0.0, pending_rate=0.0)
    )
    fs.plan(fleet_env.fleet)
    fs.collect(fleet_env.fleet)

    with pytest.raises(RolloutHalted) as caught:
        fs.advance(fleet_env.fleet)
    report.note(str(caught.value))


def test_응답이_충분히_안_왔으면_넘어가지_않는다(fleet_env) -> None:
    """응답 없이 넘어가면 **단계를 나눈 의미가 없다.**"""
    fleet_env.fleet.ota.set_profile(
        FleetResponseProfile(failure_rate=0.0, offline_rate=0.0, pending_rate=0.9)
    )
    fs.plan(fleet_env.fleet)
    view = fs.collect(fleet_env.fleet)
    report.block("대부분 아직 답이 없다", view.render())

    with pytest.raises(IllegalStateTransition) as caught:
        fs.advance(fleet_env.fleet)
    assert "OTA_NOT_ENOUGH_REPORTED" in str(caught.value)
    report.note(str(caught.value)[:150])


def test_단계는_커지는_순서여야_한다() -> None:
    with pytest.raises(InvariantViolation) as caught:
        Rollout.plan(
            RolloutId.of("r"), "v2", (wave("big", 20), wave("small", 2, start=20))
        )
    report.note(str(caught.value))
    report.note("큰 단계를 먼저 하면 작은 단계에서 확인할 것이 없다.")


def test_첫_단계는_전체의_10퍼센트_이하여야_한다() -> None:
    with pytest.raises(InvariantViolation) as caught:
        Rollout.plan(
            RolloutId.of("r"), "v2", (wave("w1", 15), wave("w2", 20, start=15))
        )
    report.note(str(caught.value))
    report.note("**문제가 나도 그만큼만 겪는다** — 그게 canary 의 정의다.")


def test_같은_디바이스가_두_단계에_들어가면_막는다() -> None:
    with pytest.raises(InvariantViolation) as caught:
        Rollout.plan(
            RolloutId.of("r"),
            "v2",
            (wave("w1", 2), wave("w2", 20)),  # DEV-00, DEV-01 이 겹친다
        )
    report.note(str(caught.value))
    report.note("**같은 디바이스를 두 번 세면 실패율이 거짓말을 한다.**")


def test_대상이_아닌_디바이스의_결과는_받지_않는다() -> None:
    rollout = Rollout.plan(
        RolloutId.of("r"), "v2", (wave("w1", 2), wave("w2", 20, start=2))
    )
    rollout.start("2026-05-24 12:00:00")
    with pytest.raises(InvariantViolation):
        rollout.record_wave(
            {"DEV-99": DeviceOutcome.SUCCEEDED},
            RolloutPolicy(),
            "2026-05-24 12:30:00",
        )
    report.note("잘못 들어온 보고가 단계 결과를 오염시키지 않는다.")


def test_사람이_직접_멈출_수도_있다(fleet_env) -> None:
    fs.plan(fleet_env.fleet)
    fs.collect(fleet_env.fleet)

    from application.fleet.release_and_rollout import HaltRolloutCommand

    view = fleet_env.fleet.halt_rollout().execute(
        HaltRolloutCommand(
            rollout_id="ro-1",
            reason="현장에서 알람이 늘었다는 연락",
            occurred_at="2026-05-24 14:00:00",
        )
    )
    assert view.status == "HALTED"
    assert view.halt_reason
    report.note(
        "지표가 아직 안 넘었어도 멈출 수 있다. **현장 연락이 지표보다 빠를 때가 있다.**"
    )


def test_실제_IoT_Job_이_만들어진다(fleet_env) -> None:
    """알리는 것까지는 **진짜다.** moto 안이지만 진짜 boto3 호출이다."""
    import boto3

    fs.plan(fleet_env.fleet)
    iot = boto3.client("iot", region_name="ap-northeast-2")
    jobs = [j["jobId"] for j in iot.list_jobs().get("jobs", [])]

    report.block("만들어진 IoT Job", "\n".join(f"  {j}" for j in jobs))
    assert any(j.startswith("ota-ro-1-") for j in jobs)

    detail = iot.get_job_document(jobId=jobs[0])
    report.note(
        "Job 문서에는 **디바이스가 스스로 검증할 수 있는 것**이 들어 있다 — "
        "위치, 체크섬, 크기, 그리고 전처리."
    )
    assert "checksum" in detail["document"]
    assert "normalization" in detail["document"]
