"""Fleet Context 의 Domain Event.

이 Context 의 Event 는 **양쪽으로** 나간다.

    사람에게   롤아웃 중단, 되돌림 — 현장이 즉시 알아야 한다
    시스템에게 릴리스 등록, 채널 승격 — 다음 단계가 이것을 기다린다
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.fleet.identifiers import FleetId, RolloutId
from domain.shared.events import DomainEvent


@dataclass(frozen=True, slots=True)
class DeviceRegistered(DomainEvent):
    fleet_id: FleetId
    device_id: str
    group: str


@dataclass(frozen=True, slots=True)
class DeviceVersionChanged(DomainEvent):
    """디바이스가 **스스로 말한** 버전이 바뀌었다.

    서버가 보냈다고 바뀌는 것이 아니다. 디바이스가 그렇다고 해야 바뀐다.
    """

    fleet_id: FleetId
    device_id: str
    from_version: str
    to_version: str


@dataclass(frozen=True, slots=True)
class ReleasePublished(DomainEvent):
    fleet_id: FleetId
    version: str
    channel: str
    artifact_bytes: int


@dataclass(frozen=True, slots=True)
class ReleasePromoted(DomainEvent):
    fleet_id: FleetId
    version: str
    channel: str
    displaced: str


@dataclass(frozen=True, slots=True)
class RolloutPlanned(DomainEvent):
    rollout_id: RolloutId
    version: str
    wave_count: int
    device_count: int


@dataclass(frozen=True, slots=True)
class WaveStarted(DomainEvent):
    rollout_id: RolloutId
    wave: str
    device_count: int


@dataclass(frozen=True, slots=True)
class RolloutHaltedEvent(DomainEvent):
    """현장이 즉시 알아야 하는 사건."""

    rollout_id: RolloutId
    wave: str
    reason: str


@dataclass(frozen=True, slots=True)
class RolloutCompleted(DomainEvent):
    rollout_id: RolloutId
    version: str
    succeeded: int
    total: int


@dataclass(frozen=True, slots=True)
class RolloutRolledBack(DomainEvent):
    rollout_id: RolloutId
    version: str
    to_version: str
    reason: str


@dataclass(frozen=True, slots=True)
class TrainingJobSubmitted(DomainEvent):
    job_id: str
    dataset_uri: str
    instance_type: str


@dataclass(frozen=True, slots=True)
class TrainingJobFinished(DomainEvent):
    job_id: str
    status: str
    artifact_uri: str
