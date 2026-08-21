"""FleetRepository / RolloutRepository 의 인메모리 구현."""

from __future__ import annotations

from collections.abc import Sequence

from domain.fleet.fleet import Fleet
from domain.fleet.identifiers import FleetId, RolloutId
from domain.fleet.rollout import Rollout


class InMemoryFleetRepository:
    """domain.fleet.ports.FleetRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, Fleet] = {}

    def save(self, fleet: Fleet) -> None:
        self._items[str(fleet.id)] = fleet

    def find_by_id(self, fleet_id: FleetId) -> Fleet | None:
        return self._items.get(str(fleet_id))

    def exists(self, fleet_id: FleetId) -> bool:
        return str(fleet_id) in self._items

    def list_all(self) -> Sequence[Fleet]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()


class InMemoryRolloutRepository:
    """domain.fleet.ports.RolloutRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, Rollout] = {}

    def save(self, rollout: Rollout) -> None:
        self._items[str(rollout.id)] = rollout

    def find_by_id(self, rollout_id: RolloutId) -> Rollout | None:
        return self._items.get(str(rollout_id))

    def exists(self, rollout_id: RolloutId) -> bool:
        return str(rollout_id) in self._items

    def list_all(self) -> Sequence[Rollout]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()
