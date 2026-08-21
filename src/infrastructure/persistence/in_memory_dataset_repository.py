"""DatasetRepository 의 인메모리 구현.

교육 초반에는 DB 가 본질이 아니다. Domain 이 먼저 서야 한다. (CLAUDE.md §4)
나중에 SQLAlchemy / DynamoDB 구현으로 갈아끼워도 Application 은 바뀌지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence

from domain.data.dataset import Dataset
from domain.data.identifiers import DatasetId


class InMemoryDatasetRepository:
    """domain.data.ports.DatasetRepository 구현."""

    def __init__(self) -> None:
        self._items: dict[str, Dataset] = {}

    def save(self, dataset: Dataset) -> None:
        self._items[str(dataset.id)] = dataset

    def find_by_id(self, dataset_id: DatasetId) -> Dataset | None:
        return self._items.get(str(dataset_id))

    def exists(self, dataset_id: DatasetId) -> bool:
        return str(dataset_id) in self._items

    def list_all(self) -> Sequence[Dataset]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()
