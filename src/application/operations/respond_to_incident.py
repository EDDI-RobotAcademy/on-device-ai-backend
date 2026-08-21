"""문제가 발생하면 격리하고, 필요하면 되돌린다. (실습 5-8, 5-10)

두 동작의 차이를 이 파일이 보여준다.

    Quarantine  판단을 멈춘다. 모델은 그대로 있다.
    Rollback    이전 버전으로 되돌린다. **새 버전이 생긴다.**

격리가 먼저인 이유: 이전 모델이 더 나으리라는 보장이 없다.
입력이 변한 것이라면(실습 5-7) 이전 모델도 똑같이 틀린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.operations.dto import DeploymentView, IncidentView
from application.operations.support import (
    commit,
    load_deployment,
    moment,
    watch_for,
)
from application.shared.errors import ConflictingRequest
from application.shared.ports import Clock, EventPublisher
from domain.operations.identifiers import IncidentId
from domain.operations.incident import IncidentPolicy
from domain.operations.ports import DeploymentRepository, HealthWatchRepository


@dataclass(frozen=True, slots=True)
class QuarantineCommand:
    deployment_id: str
    reason: str = ""
    """비우면 최근 관측의 CRITICAL 소견을 이유로 쓴다."""

    occurred_at: str | None = None
    policy: IncidentPolicy = field(default_factory=IncidentPolicy)


class QuarantineDeployment:
    """판단을 멈춘다. (실습 5-8)

    이유를 안 주면 **최근 관측에서 이유를 찾아 온다.**
    사람이 이유를 지어내는 것보다 기계가 근거를 붙이는 편이 정확하다.
    """

    def __init__(
        self,
        deployments: DeploymentRepository,
        watches: HealthWatchRepository,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._deployments = deployments
        self._watches = watches
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: QuarantineCommand) -> DeploymentView:
        deployment = load_deployment(self._deployments, command.deployment_id)
        reason = command.reason.strip()

        if not reason:
            watch = watch_for(self._watches, deployment.id)
            should, derived = command.policy.should_quarantine(watch.latest)
            if not should:
                raise ConflictingRequest(
                    "최근 관측에 격리할 근거가 없다. "
                    "그래도 멈추려면 이유를 직접 적어야 한다.",
                    subject=str(deployment.id),
                )
            reason = derived

        deployment.quarantine(reason, moment(self._clock, command.occurred_at))
        commit(self._deployments, deployment, self._publisher)
        return DeploymentView.of(deployment)


@dataclass(frozen=True, slots=True)
class ResumeCommand:
    deployment_id: str
    reason: str
    occurred_at: str | None = None


class ResumeDeployment:
    def __init__(
        self,
        deployments: DeploymentRepository,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._deployments = deployments
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: ResumeCommand) -> DeploymentView:
        deployment = load_deployment(self._deployments, command.deployment_id)
        deployment.resume(command.reason, moment(self._clock, command.occurred_at))
        commit(self._deployments, deployment, self._publisher)
        return DeploymentView.of(deployment)


@dataclass(frozen=True, slots=True)
class RollbackCommand:
    """망가진 모델을 Rollback하라. (실습 5-10)"""

    deployment_id: str
    to_version: int
    reason: str
    occurred_at: str | None = None


class RollbackDeployment:
    def __init__(
        self,
        deployments: DeploymentRepository,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._deployments = deployments
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: RollbackCommand) -> DeploymentView:
        deployment = load_deployment(self._deployments, command.deployment_id)
        deployment.rollback(
            command.to_version,
            command.reason,
            moment(self._clock, command.occurred_at),
        )
        commit(self._deployments, deployment, self._publisher)
        return DeploymentView.of(deployment)


@dataclass(frozen=True, slots=True)
class ResolveIncidentCommand:
    watch_id: str
    incident_id: str
    resolution: str


class ResolveIncident:
    def __init__(
        self,
        watches: HealthWatchRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._watches = watches
        self._publisher = publisher

    def execute(self, command: ResolveIncidentCommand) -> IncidentView:
        from application.operations.support import load_watch

        watch = load_watch(self._watches, command.watch_id)
        incident = watch.resolve_incident(
            IncidentId.of(command.incident_id), command.resolution
        )
        commit(self._watches, watch, self._publisher)
        return IncidentView.of(str(watch.id), incident)


@dataclass(frozen=True, slots=True)
class ListIncidentsQuery:
    watch_id: str
    only_open: bool = False


class ListIncidents:
    def __init__(self, watches: HealthWatchRepository) -> None:
        self._watches = watches

    def execute(self, query: ListIncidentsQuery) -> tuple[IncidentView, ...]:
        from application.operations.support import load_watch

        watch = load_watch(self._watches, query.watch_id)
        incidents = watch.open_incidents if query.only_open else watch.incidents
        return tuple(IncidentView.of(str(watch.id), i) for i in incidents)
