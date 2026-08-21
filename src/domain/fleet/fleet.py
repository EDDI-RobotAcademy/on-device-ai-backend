"""Fleet — 수천 대를 하나로. (Aggregate Root, 실습 6-3, 6-7, 6-11)

이 Aggregate 가 들고 있는 것은 셋이다.

    디바이스 목록   누가 있고, 무엇을 돌리고 있고, 언제 마지막으로 봤는가
    채널 상태       지금 canary 와 stable 이 무엇인가
    계보            어느 릴리스가 어느 데이터에서 왔는가 (실습 6-10)

Aggregate 하나가 3,000대를 들고 있는 것이 옳은가?
**여기서는 옳다.** 배포 판정이 전체 상태를 봐야 하기 때문이다 —
"몇 대가 어느 버전인가"를 모르면 다음 wave 를 정할 수 없다.

다만 이 사실이 규모의 상한을 정한다. 수십만 대가 되면 나눠야 한다 —
그때는 Fleet 을 사이트별로 쪼개고, 집계를 읽기 전용 모델로 뺀다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from domain.fleet import events as domain_events
from domain.fleet.device import Device, DeviceStatus, FleetSummary
from domain.fleet.errors import DeviceNotFound, NotReleasable
from domain.fleet.identifiers import FleetId
from domain.fleet.release import ChannelState, ReleaseBundle, ReleaseChannel
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.events import EventRecorder


class Fleet(EventRecorder):
    """디바이스 집합 하나."""

    __slots__ = ("_id", "_name", "_devices", "_channels", "_releases", "_lineage")

    def __init__(self, fleet_id: FleetId, name: str) -> None:
        super().__init__()
        self._id = fleet_id
        self._name = name
        self._devices: dict[str, Device] = {}
        self._channels = ChannelState()
        self._releases: dict[str, ReleaseBundle] = {}
        self._lineage: dict[str, tuple[str, str]] = {}

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def create(cls, fleet_id: FleetId, name: str) -> Fleet:
        if not name.strip():
            raise InvariantViolation("이름이 없다.", subject="name")
        return cls(fleet_id, name.strip())

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> FleetId:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def devices(self) -> tuple[Device, ...]:
        return tuple(self._devices.values())

    @property
    def size(self) -> int:
        return len(self._devices)

    @property
    def channels(self) -> ChannelState:
        return self._channels

    @property
    def releases(self) -> tuple[ReleaseBundle, ...]:
        return tuple(self._releases.values())

    def device(self, device_id: str) -> Device:
        found = self._devices.get(device_id)
        if found is None:
            raise DeviceNotFound(
                f"디바이스 '{device_id}' 가 이 플릿에 없다.", subject=device_id
            )
        return found

    def release_of(self, version: str) -> ReleaseBundle | None:
        return self._releases.get(version)

    def devices_in(self, group: str) -> tuple[Device, ...]:
        return tuple(d for d in self._devices.values() if d.group == group)

    def devices_on(self, version: str) -> tuple[Device, ...]:
        return tuple(d for d in self._devices.values() if d.current_version == version)

    def reachable_devices(self, group: str | None = None) -> tuple[Device, ...]:
        """지금 내보내면 받을 가능성이 있는 대상.

        **격리된 디바이스는 여기 안 들어온다.** 모듈 5 가 판단을 멈춰 세웠기 때문이다.
        """
        pool = self.devices_in(group) if group else self.devices
        return tuple(d for d in pool if d.is_reachable)

    def trainable_devices(self) -> tuple[Device, ...]:
        """이 디바이스들의 데이터는 학습에 써도 된다. (실습 6-4)"""
        return tuple(d for d in self._devices.values() if d.is_trainable_source)

    @property
    def summary(self) -> FleetSummary:
        """3,000대를 여섯 줄로. **목록이 아니라 집계다.**"""
        by_status: dict[str, int] = {}
        by_version: dict[str, int] = {}
        by_group: dict[str, int] = {}
        never = 0
        for device in self._devices.values():
            by_status[device.status.value] = by_status.get(device.status.value, 0) + 1
            by_version[device.current_version] = (
                by_version.get(device.current_version, 0) + 1
            )
            by_group[device.group] = by_group.get(device.group, 0) + 1
            if not device.has_reported:
                never += 1
        return FleetSummary(
            total=len(self._devices),
            by_status=by_status,
            by_version=by_version,
            by_group=by_group,
            never_reported=never,
        )

    def lineage_of(self, version: str) -> tuple[str, str]:
        """이 릴리스가 어느 데이터셋·어느 학습에서 왔는가. (실습 6-10)"""
        return self._lineage.get(version, ("", ""))

    def render(self) -> str:
        lines = [
            f"플릿 {self._id} — {self._name}",
            f"  채널: {self._channels.describe()}",
            "",
            self.summary.render(),
        ]
        return "\n".join(lines)

    # -- 행위 --------------------------------------------------------------
    def register(self, device: Device) -> None:
        """디바이스를 등록한다. (실습 6-1, 6-3)"""
        if device.device_id in self._devices:
            raise InvariantViolation(
                f"'{device.device_id}' 가 이미 등록돼 있다. "
                "같은 식별자를 두 대가 쓰면 어느 쪽 데이터인지 영영 모른다.",
                subject=device.device_id,
            )
        self._devices[device.device_id] = device
        self._record(
            domain_events.DeviceRegistered(
                fleet_id=self._id, device_id=device.device_id, group=device.group
            )
        )

    def register_many(self, devices: Iterable[Device]) -> int:
        count = 0
        for device in devices:
            self.register(device)
            count += 1
        return count

    def report(
        self,
        device_id: str,
        *,
        seen_at: str,
        version: str | None = None,
        status: DeviceStatus | None = None,
    ) -> Device:
        """디바이스가 살아 있다고 알려 왔다. (실습 6-1)

        **버전은 디바이스가 말하는 것을 믿는다.**
        서버가 "보냈으니 올라갔겠지"라고 적어 두면, 실제로 안 올라간 대수를 영영 모른다.
        """
        from dataclasses import replace

        current = self.device(device_id)
        if current.has_reported and seen_at < current.last_seen_at:
            raise InvariantViolation(
                f"'{device_id}' 의 보고가 직전보다 앞선 시각이다. "
                "디바이스 시계가 어긋났거나 재전송이다.",
                subject=device_id,
            )

        updated = replace(
            current,
            last_seen_at=seen_at,
            current_version=version if version is not None else current.current_version,
            status=status if status is not None else DeviceStatus.HEALTHY,
        )
        self._devices[device_id] = updated
        if version and version != current.current_version:
            self._record(
                domain_events.DeviceVersionChanged(
                    fleet_id=self._id,
                    device_id=device_id,
                    from_version=current.current_version,
                    to_version=version,
                )
            )
        return updated

    def mark(self, device_id: str, status: DeviceStatus, note: str = "") -> Device:
        """상태를 바꾼다. 격리·정비·폐기가 여기로 들어온다."""
        from dataclasses import replace

        current = self.device(device_id)
        updated = replace(current, status=status, note=note or current.note)
        self._devices[device_id] = updated
        return updated

    def sweep_stale(self, *, now: str, stale_after: str, unreachable_after: str) -> int:
        """오래 연락 없는 디바이스를 표시한다. (실습 6-11)

        시각 비교만 한다. **Domain 은 시계를 모른다** — 기준 시각을 받아서 쓴다.
        """
        from dataclasses import replace

        changed = 0
        for device_id, device in list(self._devices.items()):
            if device.status in (DeviceStatus.QUARANTINED, DeviceStatus.RETIRED):
                continue
            if not device.has_reported:
                continue
            if device.last_seen_at < unreachable_after:
                new_status = DeviceStatus.UNREACHABLE
            elif device.last_seen_at < stale_after:
                new_status = DeviceStatus.STALE
            else:
                new_status = DeviceStatus.HEALTHY
            if new_status is not device.status:
                self._devices[device_id] = replace(device, status=new_status)
                changed += 1
        return changed

    def publish(self, bundle: ReleaseBundle, build_id: str = "", job_id: str = "") -> None:
        """릴리스를 이 플릿에 등록한다. (실습 6-6)

        등록만 한다. 아직 아무 디바이스에도 안 갔다 — 그것은 롤아웃의 일이다.
        """
        if bundle.version in self._releases:
            raise InvariantViolation(
                f"버전 '{bundle.version}' 이 이미 있다. "
                "같은 버전 이름으로 다른 내용을 올리면 계보가 끊긴다.",
                subject=bundle.version,
            )
        self._releases[bundle.version] = bundle
        self._lineage[bundle.version] = (
            build_id or bundle.source_build_id,
            job_id or bundle.source_job_id,
        )
        self._record(
            domain_events.ReleasePublished(
                fleet_id=self._id,
                version=bundle.version,
                channel=bundle.channel.value,
                artifact_bytes=bundle.artifact_bytes,
            )
        )

    def promote(self, version: str, channel: ReleaseChannel) -> ChannelState:
        """채널에 올린다. (실습 6-7)

        **한 채널에 하나뿐이다.** 있던 것은 ARCHIVED 로 밀려난다.
        그리고 STABLE 로 가려면 CANARY 를 거쳐야 한다.
        """
        bundle = self._releases.get(version)
        if bundle is None:
            raise NotReleasable(
                f"'{version}' 은 이 플릿에 등록된 적이 없다.", subject=version
            )
        if channel is ReleaseChannel.STABLE and self._channels.canary != version:
            raise IllegalStateTransition(
                f"'{version}' 은 canary 를 거치지 않았다. "
                "몇 대에서 확인하지 않은 것을 전부에게 보내지 않는다.",
                subject=version,
            )

        previous = self._channels.current(channel)
        self._channels = self._channels.with_promotion(version, channel)
        self._record(
            domain_events.ReleasePromoted(
                fleet_id=self._id,
                version=version,
                channel=channel.value,
                displaced=previous,
            )
        )
        return self._channels

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return f"Fleet(id={self._id}, devices={len(self._devices)})"


def summarize(devices: Iterable[Device]) -> Mapping[str, int]:
    """버전별 대수. 배포 계획을 짤 때 가장 먼저 보는 숫자다."""
    counts: dict[str, int] = {}
    for device in devices:
        counts[device.current_version] = counts.get(device.current_version, 0) + 1
    return counts
