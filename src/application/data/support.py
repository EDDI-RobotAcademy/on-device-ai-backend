"""Use Case 들이 공유하는 최소한의 거들기.

여기에 판단을 넣지 않는다. 조회 실패 처리와 Event 발행만 담는다.
"""

from __future__ import annotations

from application.shared.ports import EventPublisher
from domain.data.dataset import Dataset
from domain.data.errors import DatasetNotFound
from domain.data.identifiers import DatasetId
from domain.data.ports import DatasetRepository


def load_dataset(repository: DatasetRepository, dataset_id: str | DatasetId) -> Dataset:
    identifier = (
        dataset_id if isinstance(dataset_id, DatasetId) else DatasetId.of(dataset_id)
    )
    dataset = repository.find_by_id(identifier)
    if dataset is None:
        raise DatasetNotFound(
            f"Dataset '{identifier}' 이 등록되어 있지 않다.", subject=str(identifier)
        )
    return dataset


def commit(
    repository: DatasetRepository,
    dataset: Dataset,
    publisher: EventPublisher | None = None,
) -> None:
    """Aggregate 를 저장하고, 그 과정에서 발생한 Event 를 내보낸다.

    저장이 먼저다. Event 만 나가고 상태가 남지 않는 상황을 만들지 않는다.
    """
    repository.save(dataset)
    events = dataset.pull_events()
    if publisher is not None and events:
        publisher.publish(events)
