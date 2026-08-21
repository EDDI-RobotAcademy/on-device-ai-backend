"""GetDataset / ListDatasets — 조회.

조회는 Use Case 가 아니라고 보는 시각도 있지만,
Interface Layer 가 Repository 를 직접 만지기 시작하면 의존 방향이 무너진다.
얇게라도 여기를 지난다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import DatasetView
from application.data.support import load_dataset
from domain.data.ports import DatasetRepository


@dataclass(frozen=True, slots=True)
class GetDatasetQuery:
    dataset_id: str


class GetDataset:
    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    def execute(self, query: GetDatasetQuery) -> DatasetView:
        return DatasetView.of(load_dataset(self._repository, query.dataset_id))


class ListDatasets:
    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    def execute(self) -> tuple[DatasetView, ...]:
        return tuple(DatasetView.of(d) for d in self._repository.list_all())
