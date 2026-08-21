"""DeploymentRepository / HealthWatchRepository 의 인메모리 구현."""

from __future__ import annotations

from collections.abc import Sequence

from domain.operations.deployment import Deployment
from domain.operations.identifiers import DeploymentId, WatchId
from domain.operations.watch import HealthWatch


class InMemoryDeploymentRepository:
    """domain.operations.ports.DeploymentRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, Deployment] = {}

    def save(self, deployment: Deployment) -> None:
        self._items[str(deployment.id)] = deployment

    def find_by_id(self, deployment_id: DeploymentId) -> Deployment | None:
        return self._items.get(str(deployment_id))

    def exists(self, deployment_id: DeploymentId) -> bool:
        return str(deployment_id) in self._items

    def list_all(self) -> Sequence[Deployment]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()


class InMemoryHealthWatchRepository:
    """domain.operations.ports.HealthWatchRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, HealthWatch] = {}

    def save(self, watch: HealthWatch) -> None:
        self._items[str(watch.id)] = watch

    def find_by_id(self, watch_id: WatchId) -> HealthWatch | None:
        return self._items.get(str(watch_id))

    def find_by_deployment(self, deployment_id: DeploymentId) -> HealthWatch | None:
        return next(
            (w for w in self._items.values() if w.deployment_id == deployment_id),
            None,
        )

    def exists(self, watch_id: WatchId) -> bool:
        return str(watch_id) in self._items

    def list_all(self) -> Sequence[HealthWatch]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()
