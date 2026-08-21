"""Rollout Aggregate 의 불변식.

AWS 없이 돌아간다. **이 파일이 §15 의 증거다** — moto 도 boto3 도 필요 없다.
"""

from __future__ import annotations

import pytest

from domain.fleet.errors import RolloutHalted
from domain.fleet.identifiers import RolloutId
from domain.fleet.rollout import (
    DeviceOutcome,
    Rollout,
    RolloutPolicy,
    RolloutStatus,
    Wave,
    WaveResult,
)
from domain.shared.errors import IllegalStateTransition, InvariantViolation

AT = "2026-05-24 12:00:00"


def wave(name: str, count: int, start: int = 0) -> Wave:
    return Wave(
        name=name, device_ids=tuple(f"DEV-{n:02d}" for n in range(start, start + count))
    )


def planned(*, waves=None) -> Rollout:  # noqa: ANN001
    return Rollout.plan(
        RolloutId.of("ro-1"),
        "v2.0.0",
        waves or (wave("w1", 2), wave("w2", 8, start=2), wave("w3", 14, start=10)),
        previous_version="v1.0.0",
    )


def running() -> Rollout:
    rollout = planned()
    rollout.start(AT)
    return rollout


def all_ok(target: Wave) -> dict[str, DeviceOutcome]:
    return dict.fromkeys(target.device_ids, DeviceOutcome.SUCCEEDED)


class Test계획:
    def test_단계가_없으면_그냥_던지는_것이다(self) -> None:
        with pytest.raises(InvariantViolation):
            Rollout.plan(RolloutId.of("r"), "v2", ())

    def test_단계는_커지는_순서여야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            Rollout.plan(
                RolloutId.of("r"), "v2", (wave("a", 20), wave("b", 2, start=20))
            )

    def test_첫_단계는_작아야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            Rollout.plan(
                RolloutId.of("r"), "v2", (wave("a", 15), wave("b", 20, start=15))
            )

    def test_작은_플릿에서도_최소_대수는_허용한다(self) -> None:
        """10% 로만 재면 8대짜리 플릿의 첫 단계가 0대가 된다."""
        rollout = Rollout.plan(
            RolloutId.of("r"), "v2", (wave("a", 2), wave("b", 6, start=2))
        )
        assert rollout.device_count == 8

    def test_같은_디바이스가_두_단계에_들어갈_수_없다(self) -> None:
        with pytest.raises(InvariantViolation) as caught:
            Rollout.plan(RolloutId.of("r"), "v2", (wave("a", 2), wave("b", 20)))
        assert "거짓말" in str(caught.value)

    def test_계획하면_사건이_남는다(self) -> None:
        events = planned().pull_events()
        assert [e.event_name for e in events] == ["RolloutPlanned"]


class Test진행:
    def test_시작은_한_번뿐이다(self) -> None:
        rollout = running()
        with pytest.raises(IllegalStateTransition):
            rollout.start(AT)

    def test_결과_없이_다음_단계로_못_간다(self) -> None:
        rollout = running()
        with pytest.raises(IllegalStateTransition):
            rollout.advance(RolloutPolicy(), AT)

    def test_대상이_아닌_디바이스의_결과는_거부한다(self) -> None:
        rollout = running()
        with pytest.raises(InvariantViolation):
            rollout.record_wave(
                {"DEV-99": DeviceOutcome.SUCCEEDED}, RolloutPolicy(), AT
            )

    def test_안_적힌_디바이스는_PENDING_이_된다(self) -> None:
        rollout = running()
        result = rollout.record_wave(
            {"DEV-00": DeviceOutcome.SUCCEEDED}, RolloutPolicy(), AT
        )
        assert result.outcomes["DEV-01"] is DeviceOutcome.PENDING

    def test_단계를_순서대로_지난다(self) -> None:
        rollout = running()
        rollout.record_wave(all_ok(rollout.current_wave), RolloutPolicy(), AT)
        second = rollout.advance(RolloutPolicy(), AT)
        assert second.name == "w2"

        rollout.record_wave(all_ok(second), RolloutPolicy(), AT)
        third = rollout.advance(RolloutPolicy(), AT)
        assert third.name == "w3"

    def test_마지막_단계_뒤에는_완료된다(self) -> None:
        rollout = running()
        for _ in range(3):
            wave_now = rollout.current_wave
            rollout.record_wave(all_ok(wave_now), RolloutPolicy(), AT)
            if wave_now.name != "w3":
                rollout.advance(RolloutPolicy(), AT)

        rollout.complete(AT)
        assert rollout.status is RolloutStatus.COMPLETED
        assert rollout.coverage == 1.0


class Test실패:
    def test_실패율의_분모는_시도한_것이다(self) -> None:
        result = WaveResult(
            wave=wave("w", 6),
            outcomes={
                "DEV-00": DeviceOutcome.SUCCEEDED,
                "DEV-01": DeviceOutcome.SUCCEEDED,
                "DEV-02": DeviceOutcome.FAILED,
                "DEV-03": DeviceOutcome.UNREACHABLE,
                "DEV-04": DeviceOutcome.UNREACHABLE,
                "DEV-05": DeviceOutcome.PENDING,
            },
        )
        assert result.attempted == 3
        assert result.failure_ratio == pytest.approx(1 / 3)
        assert result.reported == 5

    def test_실패가_기준을_넘으면_스스로_멈춘다(self) -> None:
        rollout = running()
        rollout.record_wave(
            {"DEV-00": DeviceOutcome.SUCCEEDED, "DEV-01": DeviceOutcome.FAILED},
            RolloutPolicy(max_failure_ratio=0.05),
            AT,
        )
        assert rollout.status is RolloutStatus.HALTED
        assert rollout.halt_reason

    def test_멈춘_뒤에는_진행할_수_없다(self) -> None:
        rollout = running()
        rollout.record_wave(
            {"DEV-00": DeviceOutcome.FAILED, "DEV-01": DeviceOutcome.FAILED},
            RolloutPolicy(),
            AT,
        )
        with pytest.raises(RolloutHalted):
            rollout.advance(RolloutPolicy(), AT)

    def test_멈춘_뒤에는_결과도_못_적는다(self) -> None:
        rollout = running()
        rollout.record_wave(
            {"DEV-00": DeviceOutcome.FAILED, "DEV-01": DeviceOutcome.FAILED},
            RolloutPolicy(),
            AT,
        )
        with pytest.raises(IllegalStateTransition):
            rollout.record_wave({}, RolloutPolicy(), AT)

    def test_응답이_적으면_다음으로_못_간다(self) -> None:
        rollout = running()
        rollout.record_wave(
            {"DEV-00": DeviceOutcome.SUCCEEDED}, RolloutPolicy(), AT
        )
        with pytest.raises(IllegalStateTransition) as caught:
            rollout.advance(RolloutPolicy(min_reported_before_advance=0.9), AT)
        assert "OTA_NOT_ENOUGH_REPORTED" in str(caught.value)

    def test_미도달이_많으면_경고한다(self) -> None:
        result = WaveResult(
            wave=wave("w", 4),
            outcomes={
                "DEV-00": DeviceOutcome.SUCCEEDED,
                "DEV-01": DeviceOutcome.UNREACHABLE,
                "DEV-02": DeviceOutcome.UNREACHABLE,
                "DEV-03": DeviceOutcome.SUCCEEDED,
            },
        )
        codes = {f.code for f in RolloutPolicy().inspect(result)}
        assert "OTA_TOO_MANY_UNREACHABLE" in codes


class Test중단과재개:
    def test_이유_없이_멈추지_않는다(self) -> None:
        with pytest.raises(InvariantViolation):
            running().halt("  ", AT)

    def test_확인한_것을_적어야_재개한다(self) -> None:
        rollout = running()
        rollout.halt("현장 연락", AT)

        with pytest.raises(InvariantViolation):
            rollout.resume("", AT)

        rollout.resume("펌웨어 버전 확인 완료", AT)
        assert rollout.status is RolloutStatus.RUNNING
        assert rollout.halt_reason == ""

    def test_멈추지_않은_것은_재개할_수_없다(self) -> None:
        with pytest.raises(IllegalStateTransition):
            running().resume("아무거나", AT)


class Test되돌리기:
    def test_되돌려졌다는_사실이_남는다(self) -> None:
        rollout = running()
        rollout.mark_rolled_back("FAULT 재현율 붕괴", AT)

        assert rollout.status is RolloutStatus.ROLLED_BACK
        events = rollout.pull_events()
        assert any(e.event_name == "RolloutRolledBack" for e in events)

    def test_이유가_필요하다(self) -> None:
        with pytest.raises(InvariantViolation):
            running().mark_rolled_back("  ", AT)

    def test_계획_상태는_되돌릴_대상이_아니다(self) -> None:
        with pytest.raises(IllegalStateTransition):
            planned().mark_rolled_back("아직 시작도 안 했다", AT)


class Test도달률:
    def test_100퍼센트는_오지_않는다(self) -> None:
        rollout = running()
        rollout.record_wave(
            {"DEV-00": DeviceOutcome.SUCCEEDED, "DEV-01": DeviceOutcome.UNREACHABLE},
            RolloutPolicy(),
            AT,
        )
        assert rollout.succeeded_count == 1
        assert rollout.unreachable_count == 1
        assert rollout.coverage < 1.0

    def test_이력이_남는다(self) -> None:
        rollout = running()
        rollout.record_wave(all_ok(rollout.current_wave), RolloutPolicy(), AT)
        rollout.halt("확인 필요", AT)

        assert len(rollout.history) >= 3
        assert any("중단" in what for _, what in rollout.history)
