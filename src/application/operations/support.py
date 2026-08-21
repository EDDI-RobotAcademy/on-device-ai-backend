"""Operations Use Case 들이 공유하는 최소한의 거들기."""

from __future__ import annotations

from application.shared.ports import Clock, EventPublisher
from domain.operations.deployment import Deployment
from domain.operations.errors import DeploymentNotFound, HealthWatchNotFound
from domain.operations.identifiers import DeploymentId, WatchId
from domain.operations.ports import DeploymentRepository, HealthWatchRepository
from domain.operations.watch import HealthWatch


def load_deployment(
    repository: DeploymentRepository, deployment_id: str | DeploymentId
) -> Deployment:
    identifier = (
        deployment_id
        if isinstance(deployment_id, DeploymentId)
        else DeploymentId.of(deployment_id)
    )
    deployment = repository.find_by_id(identifier)
    if deployment is None:
        raise DeploymentNotFound(
            f"배포 '{identifier}' 가 존재하지 않는다.", subject=str(identifier)
        )
    return deployment


def load_watch(
    repository: HealthWatchRepository, watch_id: str | WatchId
) -> HealthWatch:
    identifier = watch_id if isinstance(watch_id, WatchId) else WatchId.of(watch_id)
    watch = repository.find_by_id(identifier)
    if watch is None:
        raise HealthWatchNotFound(
            f"관측 '{identifier}' 가 존재하지 않는다.", subject=str(identifier)
        )
    return watch


def watch_for(
    repository: HealthWatchRepository, deployment_id: DeploymentId
) -> HealthWatch:
    watch = repository.find_by_deployment(deployment_id)
    if watch is None:
        raise HealthWatchNotFound(
            f"배포 '{deployment_id}' 에 대한 관측이 없다. "
            "배포와 동시에 관측을 시작하지 않으면 그 사이 구간은 영영 비어 있다.",
            subject=str(deployment_id),
        )
    return watch


def commit(
    repository, aggregate, publisher: EventPublisher | None = None  # noqa: ANN001
) -> None:
    repository.save(aggregate)
    events = aggregate.pull_events()
    if publisher is not None and events:
        publisher.publish(events)


def moment(clock: Clock, override: str | None = None) -> str:
    """이 일이 '언제' 일어난 것으로 기록할 것인가.

    기본은 지금이다. 그런데 운영 기록은 **데이터의 시각**을 따라야 할 때가 있다.
    지난 로그를 다시 넣거나(backfill), 실습에서 4일치를 몇 초에 재생할 때가 그렇다.
    그래서 호출자가 시각을 못박을 수 있게 열어 둔다.

    형식은 현장 로그와 같은 'YYYY-MM-DD HH:MM:SS' 로 맞춘다 —
    시각을 문자열로 비교하기 때문에 형식이 섞이면 순서가 뒤집힌다.
    """
    if override:
        return override
    return clock.now().replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
