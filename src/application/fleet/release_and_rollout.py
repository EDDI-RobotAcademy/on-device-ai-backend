"""Cloud에서 만든 모델을 Edge로 보내고, OTA로 업데이트하고, 되돌린다.
(실습 6-6 ~ 6-10)

순환의 아래쪽 반이다.

    결과물 → 릴리스 번들 → 채널 → 롤아웃 → 디바이스
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.fleet.dto import (
    LineageView,
    ReleaseView,
    RolloutView,
    WaveView,
)
from application.fleet.support import commit, load_fleet, load_rollout, moment
from application.shared.errors import ConflictingRequest
from application.shared.ports import Clock, EventPublisher
from domain.fleet.device import Device
from domain.fleet.identifiers import RolloutId
from domain.fleet.lineage import LineagePolicy, trace_of
from domain.fleet.ports import FleetRepository, OtaGateway, RolloutRepository
from domain.fleet.release import ReleaseBundle, ReleaseChannel, ReleasePolicy
from domain.fleet.rollout import (
    DeviceOutcome,
    Rollout,
    RolloutPolicy,
    Wave,
)


@dataclass(frozen=True, slots=True)
class PublishReleaseCommand:
    """Cloud에서 만든 모델을 Edge로 보낼 수 있게 묶는다. (실습 6-6)"""

    fleet_id: str
    bundle: ReleaseBundle
    policy: ReleasePolicy = field(default_factory=ReleasePolicy)


class PublishRelease:
    def __init__(
        self, fleets: FleetRepository, publisher: EventPublisher | None = None
    ) -> None:
        self._fleets = fleets
        self._publisher = publisher

    def execute(self, command: PublishReleaseCommand) -> ReleaseView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        check = command.policy.inspect(command.bundle, device_count=fleet.size)

        if check.can_publish:
            fleet.publish(command.bundle)
            commit(self._fleets, fleet, self._publisher)

        return ReleaseView.of(str(fleet.id), check, device_count=fleet.size)


@dataclass(frozen=True, slots=True)
class PromoteReleaseCommand:
    """모델 Version을 Cloud에서 관리하라. (실습 6-7)"""

    fleet_id: str
    version: str
    channel: ReleaseChannel


class PromoteRelease:
    def __init__(
        self, fleets: FleetRepository, publisher: EventPublisher | None = None
    ) -> None:
        self._fleets = fleets
        self._publisher = publisher

    def execute(self, command: PromoteReleaseCommand) -> str:
        fleet = load_fleet(self._fleets, command.fleet_id)
        state = fleet.promote(command.version, command.channel)
        commit(self._fleets, fleet, self._publisher)
        return state.describe()


@dataclass(frozen=True, slots=True)
class PlanRolloutCommand:
    """OTA로 현장의 AI를 업데이트하라. (실습 6-8)"""

    fleet_id: str
    rollout_id: str
    version: str
    wave_sizes: tuple[int, ...] = (2, 8, 9999)
    """단계 크기. 마지막은 남은 전부를 뜻한다."""

    group_order: tuple[str, ...] = ()
    policy: RolloutPolicy = field(default_factory=RolloutPolicy)
    occurred_at: str | None = None


class PlanRollout:
    """단계를 짜고 시작한다.

    **대상은 지금 받을 수 있는 디바이스뿐이다.**
    격리된 디바이스는 처음부터 대상이 아니다 — 실패로 세면 실패율이 거짓말을 한다.
    """

    def __init__(
        self,
        fleets: FleetRepository,
        rollouts: RolloutRepository,
        gateway: OtaGateway,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._rollouts = rollouts
        self._gateway = gateway
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: PlanRolloutCommand) -> RolloutView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        bundle = fleet.release_of(command.version)
        if bundle is None:
            from domain.fleet.errors import ReleaseNotFound

            raise ReleaseNotFound(
                f"'{command.version}' 은 이 플릿에 등록된 적이 없다.",
                subject=command.version,
            )

        targets = self._ordered_targets(fleet, command.group_order)
        if not targets:
            raise ConflictingRequest(
                "지금 받을 수 있는 디바이스가 하나도 없다.", subject=command.fleet_id
            )

        waves = _slice_waves(targets, command.wave_sizes)
        previous = fleet.channels.stable or _dominant_version(fleet)

        rollout = Rollout.plan(
            RolloutId.of(command.rollout_id),
            command.version,
            waves,
            previous_version=previous,
            policy=command.policy,
        )
        wave = rollout.start(moment(self._clock, command.occurred_at))
        commit(self._rollouts, rollout, self._publisher)

        self._gateway.announce(rollout.id, bundle, wave.device_ids)
        return RolloutView.of(rollout)

    def _ordered_targets(self, fleet, group_order: tuple[str, ...]) -> list[str]:  # noqa: ANN001
        reachable: list[Device] = list(fleet.reachable_devices())
        if not group_order:
            return [d.device_id for d in reachable]
        rank = {group: index for index, group in enumerate(group_order)}
        reachable.sort(key=lambda d: (rank.get(d.group, len(rank)), d.device_id))
        return [d.device_id for d in reachable]


@dataclass(frozen=True, slots=True)
class CollectWaveCommand:
    rollout_id: str
    policy: RolloutPolicy = field(default_factory=RolloutPolicy)
    occurred_at: str | None = None


class CollectWave:
    """디바이스들이 뭐라고 했는지 걷어 적는다. (실습 6-8)

    **응답이 없는 것은 PENDING 이다.** 실패로 세지 않는다.
    """

    def __init__(
        self,
        rollouts: RolloutRepository,
        gateway: OtaGateway,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._rollouts = rollouts
        self._gateway = gateway
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: CollectWaveCommand) -> WaveView:
        rollout = load_rollout(self._rollouts, command.rollout_id)
        wave = rollout.current_wave
        if wave is None:
            raise ConflictingRequest(
                "남은 단계가 없다.", subject=command.rollout_id
            )

        outcomes = self._gateway.collect(rollout.id, wave.device_ids)
        result = rollout.record_wave(
            outcomes, command.policy, moment(self._clock, command.occurred_at)
        )
        commit(self._rollouts, rollout, self._publisher)

        from domain.fleet.rollout import RolloutStatus

        findings = command.policy.inspect(result)
        return WaveView.of(
            str(rollout.id),
            result,
            halted=rollout.status is RolloutStatus.HALTED,
            findings=tuple(FindingView.of(f) for f in findings),
        )


@dataclass(frozen=True, slots=True)
class AdvanceRolloutCommand:
    rollout_id: str
    fleet_id: str
    policy: RolloutPolicy = field(default_factory=RolloutPolicy)
    occurred_at: str | None = None


class AdvanceRollout:
    """다음 단계로 넘어간다. (실습 6-8)"""

    def __init__(
        self,
        fleets: FleetRepository,
        rollouts: RolloutRepository,
        gateway: OtaGateway,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._rollouts = rollouts
        self._gateway = gateway
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: AdvanceRolloutCommand) -> RolloutView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        rollout = load_rollout(self._rollouts, command.rollout_id)
        bundle = fleet.release_of(rollout.version)

        wave = rollout.advance(command.policy, moment(self._clock, command.occurred_at))
        commit(self._rollouts, rollout, self._publisher)
        if bundle is not None:
            self._gateway.announce(rollout.id, bundle, wave.device_ids)
        return RolloutView.of(rollout)


@dataclass(frozen=True, slots=True)
class HaltRolloutCommand:
    rollout_id: str
    reason: str
    occurred_at: str | None = None


class HaltRollout:
    def __init__(
        self,
        rollouts: RolloutRepository,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._rollouts = rollouts
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: HaltRolloutCommand) -> RolloutView:
        rollout = load_rollout(self._rollouts, command.rollout_id)
        rollout.halt(command.reason, moment(self._clock, command.occurred_at))
        commit(self._rollouts, rollout, self._publisher)
        return RolloutView.of(rollout)


@dataclass(frozen=True, slots=True)
class RollbackRolloutCommand:
    """문제가 생기면 이전 모델로 되돌려라. (실습 6-9)"""

    fleet_id: str
    rollout_id: str
    new_rollout_id: str
    reason: str
    to_version: str = ""
    """비우면 이 롤아웃 직전 버전으로 돌아간다."""

    wave_sizes: tuple[int, ...] = (2, 8, 9999)
    policy: RolloutPolicy = field(default_factory=RolloutPolicy)
    occurred_at: str | None = None


class RollbackRollout:
    """되돌린다. **그런데 즉시 안 된다.**

    모듈 5 의 롤백은 기록 하나를 바꾸는 일이었다.
    수천 대에서는 **롤백도 또 하나의 롤아웃**이고, 똑같이 단계로 나가고
    똑같이 오프라인 디바이스를 만난다.
    """

    def __init__(
        self,
        fleets: FleetRepository,
        rollouts: RolloutRepository,
        gateway: OtaGateway,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._rollouts = rollouts
        self._gateway = gateway
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: RollbackRolloutCommand) -> RolloutView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        broken = load_rollout(self._rollouts, command.rollout_id)
        at = moment(self._clock, command.occurred_at)

        target = command.to_version or broken.previous_version
        if not target:
            raise ConflictingRequest(
                "돌아갈 버전이 없다. 첫 배포였다면 되돌릴 곳이 없다 — "
                "격리(모듈 5)가 유일한 수단이다.",
                subject=command.rollout_id,
            )
        bundle = fleet.release_of(target)
        if bundle is None:
            from domain.fleet.errors import ReleaseNotFound

            raise ReleaseNotFound(
                f"'{target}' 은 이 플릿에 등록된 적이 없다. "
                "돌아갈 곳이 없으면 롤백이 아니다.",
                subject=target,
            )

        # 새 버전을 받은 디바이스만 되돌리면 된다. **받지 않은 것은 그대로 두면 된다.**
        affected = [
            device.device_id
            for device in fleet.reachable_devices()
            if device.current_version == broken.version
        ] or [device.device_id for device in fleet.reachable_devices()]

        self._gateway.cancel(broken.id, command.reason)
        broken.mark_rolled_back(command.reason, at)
        commit(self._rollouts, broken, self._publisher)

        recovery = Rollout.plan(
            RolloutId.of(command.new_rollout_id),
            target,
            _slice_waves(affected, command.wave_sizes),
            previous_version=broken.version,
            policy=command.policy,
        )
        wave = recovery.start(at)
        commit(self._rollouts, recovery, self._publisher)
        self._gateway.announce(recovery.id, bundle, wave.device_ids)
        return RolloutView.of(recovery)


@dataclass(frozen=True, slots=True)
class ApplyOutcomesCommand:
    """디바이스가 보고한 버전을 플릿에 반영한다.

    **서버가 '보냈으니 됐겠지'라고 적지 않는다** — 디바이스가 말해야 바뀐다.
    """

    fleet_id: str
    rollout_id: str
    seen_at: str


class ApplyRolloutOutcomes:
    def __init__(
        self,
        fleets: FleetRepository,
        rollouts: RolloutRepository,
        registry,  # noqa: ANN001 - DeviceRegistry
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._rollouts = rollouts
        self._registry = registry
        self._publisher = publisher

    def execute(self, command: ApplyOutcomesCommand) -> RolloutView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        rollout = load_rollout(self._rollouts, command.rollout_id)

        for result in rollout.results:
            for device_id, outcome in result.outcomes.items():
                if outcome is DeviceOutcome.SUCCEEDED:
                    fleet.report(
                        device_id, seen_at=command.seen_at, version=rollout.version
                    )
        commit(self._fleets, fleet, self._publisher)
        for device in fleet.devices:
            self._registry.upsert(fleet.id, device)
        return RolloutView.of(rollout)


@dataclass(frozen=True, slots=True)
class TraceLineageQuery:
    """Edge → Cloud → Edge 순환 구조를 완성하라. (실습 6-10)"""

    fleet_id: str
    device_id: str
    source_devices: tuple[str, ...] = ()
    window: str = ""
    policy: LineagePolicy = field(default_factory=LineagePolicy)


class TraceLineage:
    def __init__(self, fleets: FleetRepository) -> None:
        self._fleets = fleets

    def execute(self, query: TraceLineageQuery) -> LineageView:
        fleet = load_fleet(self._fleets, query.fleet_id)
        device = fleet.device(query.device_id)
        build_id, job_id = fleet.lineage_of(device.current_version)

        trace = trace_of(
            device_id=device.device_id,
            version=device.current_version,
            job_id=job_id,
            build_id=build_id,
            window=query.window,
            detail=f"디바이스 {len(query.source_devices)}대" if query.source_devices else "",
        )
        closure = query.policy.inspect(trace, source_devices=query.source_devices)
        return LineageView.of(str(fleet.id), closure)


@dataclass(frozen=True, slots=True)
class GetRolloutQuery:
    rollout_id: str


class GetRollout:
    def __init__(self, rollouts: RolloutRepository) -> None:
        self._rollouts = rollouts

    def execute(self, query: GetRolloutQuery) -> RolloutView:
        return RolloutView.of(load_rollout(self._rollouts, query.rollout_id))


class ListRollouts:
    def __init__(self, rollouts: RolloutRepository) -> None:
        self._rollouts = rollouts

    def execute(self) -> tuple[RolloutView, ...]:
        return tuple(RolloutView.of(r) for r in self._rollouts.list_all())


# ---------------------------------------------------------------------------
def _slice_waves(device_ids: list[str], sizes: tuple[int, ...]) -> tuple[Wave, ...]:
    """대상을 단계로 자른다.

    마지막 크기는 '남은 전부'로 본다. 대수가 몇이든 계획이 성립하게 하기 위해서다.
    """
    waves: list[Wave] = []
    remaining = list(device_ids)
    for index, size in enumerate(sizes):
        if not remaining:
            break
        take = min(size, len(remaining))
        if index == len(sizes) - 1:
            take = len(remaining)
        waves.append(Wave(name=f"wave-{index + 1}", device_ids=tuple(remaining[:take])))
        remaining = remaining[take:]
    if remaining:
        waves.append(Wave(name=f"wave-{len(waves) + 1}", device_ids=tuple(remaining)))
    return tuple(waves)


def _dominant_version(fleet) -> str:  # noqa: ANN001
    return fleet.summary.dominant_version
