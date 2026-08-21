"""Operations Context 의 Domain Event.

이 Context 의 Event 는 앞의 넷과 성격이 다르다.
**대부분 사람이 받아야 하는 것들이다.** 격리와 롤백은 현장이 알아야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.operations.identifiers import DeploymentId, IncidentId, WatchId
from domain.shared.events import DomainEvent
from domain.shared.inspection import Verdict


@dataclass(frozen=True, slots=True)
class ModelDeployed(DomainEvent):
    deployment_id: DeploymentId
    target: str
    version: int
    artifact_id: str


@dataclass(frozen=True, slots=True)
class VersionReleased(DomainEvent):
    deployment_id: DeploymentId
    version: int
    artifact_id: str
    previous_version: int


@dataclass(frozen=True, slots=True)
class DeploymentQuarantined(DomainEvent):
    """현장이 즉시 알아야 하는 사건. 설비를 사람이 봐야 한다."""

    deployment_id: DeploymentId
    version: int
    reason: str


@dataclass(frozen=True, slots=True)
class DeploymentResumed(DomainEvent):
    deployment_id: DeploymentId
    version: int
    reason: str


@dataclass(frozen=True, slots=True)
class DeploymentRolledBack(DomainEvent):
    deployment_id: DeploymentId
    from_version: int
    to_version: int
    new_version: int
    reason: str


@dataclass(frozen=True, slots=True)
class HealthWatchOpened(DomainEvent):
    watch_id: WatchId
    deployment_id: DeploymentId


@dataclass(frozen=True, slots=True)
class ObservationRecorded(DomainEvent):
    watch_id: WatchId
    window_label: str
    verdict: Verdict
    sample_count: int


@dataclass(frozen=True, slots=True)
class IncidentOpened(DomainEvent):
    watch_id: WatchId
    incident_id: IncidentId
    kind: str
    window_label: str


@dataclass(frozen=True, slots=True)
class IncidentResolved(DomainEvent):
    watch_id: WatchId
    incident_id: IncidentId
    resolution: str


@dataclass(frozen=True, slots=True)
class BaselineReanchored(DomainEvent):
    """판정 기준이 바뀌었다. **판정 결과보다 중요한 사건이다.**"""

    watch_id: WatchId
    reason: str
    previous: tuple[tuple[str, float], ...]
    current: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class RetrainingRequested(DomainEvent):
    """모듈 1 로 돌아가는 신호. 순환은 여기서 닫힌다."""

    watch_id: WatchId
    deployment_id: DeploymentId
    urgency: str
    reasons: tuple[str, ...]
