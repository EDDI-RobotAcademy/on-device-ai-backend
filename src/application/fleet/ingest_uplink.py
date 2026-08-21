"""디바이스에서 발생한 데이터를 Cloud로 보내라. (실습 6-1, 6-2, 6-3)

Use Case 는 세 가지를 잇는다.

    UplinkPolicy   올려도 되는가        ← Domain 이 판단한다
    ObjectStore    어디에 둘 것인가     ← KeyLayout 이 정하고 어댑터가 실행한다
    DeviceRegistry 얼마나 올렸는가      ← 다음 검사의 입력이 된다

**거절도 결과다.** 개인정보가 섞인 묶음은 저장하지 않고, 그 사실을 돌려준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.fleet.dto import FleetView, LakeLayoutView, UplinkView
from application.fleet.support import commit, load_fleet
from application.shared.ports import EventPublisher
from domain.fleet.device import Device, DeviceStatus, FleetHealthPolicy
from domain.fleet.fleet import Fleet
from domain.fleet.identifiers import DeviceId, FleetId
from domain.fleet.object_key import KeyLayout, KeyLayoutPolicy
from domain.fleet.ports import DeviceRegistry, FleetRepository, ObjectStore
from domain.fleet.uplink import UplinkBatch, UplinkPolicy


@dataclass(frozen=True, slots=True)
class CreateFleetCommand:
    fleet_id: str
    name: str
    devices: tuple[Device, ...] = field(default_factory=tuple)


class CreateFleet:
    """플릿을 만들고 디바이스를 등록한다. (실습 6-1)"""

    def __init__(
        self,
        fleets: FleetRepository,
        registry: DeviceRegistry,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._registry = registry
        self._publisher = publisher

    def execute(self, command: CreateFleetCommand) -> FleetView:
        fleet = Fleet.create(FleetId.of(command.fleet_id), command.name)
        fleet.register_many(command.devices)
        commit(self._fleets, fleet, self._publisher)

        # 수천 대는 Aggregate 밖에도 쌓아 둔다 — 조회는 이쪽이 담당한다.
        for device in command.devices:
            self._registry.upsert(fleet.id, device)

        return FleetView.of(fleet, fleet.summary)


@dataclass(frozen=True, slots=True)
class IngestUplinkCommand:
    """디바이스가 묶음 하나를 올린다. (실습 6-1, 6-2)"""

    fleet_id: str
    batch: UplinkBatch
    body: bytes
    layout: KeyLayout = field(default_factory=KeyLayout)
    policy: UplinkPolicy = field(default_factory=UplinkPolicy)
    part: int = 0


class IngestUplink:
    def __init__(
        self,
        fleets: FleetRepository,
        store: ObjectStore,
        registry: DeviceRegistry,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._store = store
        self._registry = registry
        self._publisher = publisher

    def execute(self, command: IngestUplinkCommand) -> UplinkView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        batch = command.batch
        fleet.device(batch.device_id)  # 등록되지 않은 디바이스는 여기서 막힌다

        date = batch.window_start[:10]
        sent_today = self._registry.uplink_bytes_today(
            fleet.id, DeviceId.of(batch.device_id), date
        )
        findings = command.policy.inspect(batch, sent_today_kib=sent_today / 1024)
        accepted = command.policy.accepts(batch, sent_today_kib=sent_today / 1024)

        uri = ""
        if accepted:
            key = command.layout.key_for(
                kind=batch.kind.value.lower(),
                device_id=batch.device_id,
                date=date,
                hour=batch.window_start[11:13] or "00",
                part=command.part,
            )
            uri = self._store.put(key, command.body)
            self._registry.record_uplink(fleet.id, batch)

        # 올라왔다는 것 자체가 살아 있다는 신호다.
        fleet.report(batch.device_id, seen_at=batch.window_end)
        commit(self._fleets, fleet, self._publisher)
        self._registry.upsert(fleet.id, fleet.device(batch.device_id))

        return UplinkView(
            fleet_id=str(fleet.id),
            device_id=batch.device_id,
            accepted=accepted,
            uri=uri,
            record_count=batch.record_count,
            payload_kib=batch.payload_kib,
            sent_today_kib=(sent_today + (batch.payload_bytes if accepted else 0)) / 1024,
            findings=tuple(FindingView.of(f) for f in findings),
        )


@dataclass(frozen=True, slots=True)
class InspectLakeCommand:
    """Edge의 데이터를 S3에 모아라 — 그리고 다시 꺼낼 수 있는가. (실습 6-2)"""

    fleet_id: str
    layout: KeyLayout = field(default_factory=KeyLayout)
    policy: KeyLayoutPolicy = field(default_factory=KeyLayoutPolicy)
    filters: dict[str, str] = field(default_factory=dict)
    """좁혀 볼 조건. 이걸로 얼마나 줄어드는지가 이 실습의 요점이다."""


class InspectLakeLayout:
    def __init__(self, fleets: FleetRepository, store: ObjectStore) -> None:
        self._fleets = fleets
        self._store = store

    def execute(self, command: InspectLakeCommand) -> LakeLayoutView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        root = command.layout.prefix_for()
        stats = self._store.stats(root)

        narrowed = command.layout.prefix_for(**command.filters)
        narrowed_count = len(self._store.list_prefix(narrowed))

        findings = command.policy.inspect(command.layout, stats)
        return LakeLayoutView.of(
            str(fleet.id),
            root,
            stats,
            can_narrow=command.layout.can_narrow(**command.filters),
            narrowed_prefix=narrowed,
            narrowed_count=narrowed_count,
            findings=tuple(FindingView.of(f) for f in findings),
        )


@dataclass(frozen=True, slots=True)
class SummarizeFleetQuery:
    """수천 개의 디바이스 데이터를 하나로. (실습 6-3, 6-11)"""

    fleet_id: str
    policy: FleetHealthPolicy = field(default_factory=FleetHealthPolicy)


class SummarizeFleet:
    def __init__(self, fleets: FleetRepository) -> None:
        self._fleets = fleets

    def execute(self, query: SummarizeFleetQuery) -> FleetView:
        fleet = load_fleet(self._fleets, query.fleet_id)
        summary = fleet.summary
        findings = query.policy.inspect(summary)
        return FleetView.of(
            fleet, summary, tuple(FindingView.of(f) for f in findings)
        )


@dataclass(frozen=True, slots=True)
class SweepStaleCommand:
    """오래 연락 없는 디바이스를 표시한다. (실습 6-11)"""

    fleet_id: str
    now: str
    stale_after: str
    unreachable_after: str


class SweepStaleDevices:
    def __init__(
        self,
        fleets: FleetRepository,
        registry: DeviceRegistry,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._registry = registry
        self._publisher = publisher

    def execute(self, command: SweepStaleCommand) -> FleetView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        fleet.sweep_stale(
            now=command.now,
            stale_after=command.stale_after,
            unreachable_after=command.unreachable_after,
        )
        commit(self._fleets, fleet, self._publisher)
        for device in fleet.devices:
            self._registry.upsert(fleet.id, device)
        return FleetView.of(fleet, fleet.summary)


@dataclass(frozen=True, slots=True)
class MarkDeviceCommand:
    fleet_id: str
    device_id: str
    status: DeviceStatus
    note: str = ""


class MarkDevice:
    """격리·정비·폐기. 모듈 5 의 격리가 여기로 들어온다."""

    def __init__(
        self,
        fleets: FleetRepository,
        registry: DeviceRegistry,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._fleets = fleets
        self._registry = registry
        self._publisher = publisher

    def execute(self, command: MarkDeviceCommand) -> FleetView:
        fleet = load_fleet(self._fleets, command.fleet_id)
        device = fleet.mark(command.device_id, command.status, command.note)
        commit(self._fleets, fleet, self._publisher)
        self._registry.upsert(fleet.id, device)
        return FleetView.of(fleet, fleet.summary)
