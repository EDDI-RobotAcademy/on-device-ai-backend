"""DeclareDataSchema — 스키마를 확정하고 현실과 대조한다. (실습 1-2, 1-3)

이 Use Case 는 통과/실패를 결정하지 않는다.
Dataset 이 스스로 대조한 결과(InspectionReport)를 그대로 전달할 뿐이다.
판단 로직을 여기로 끌어오는 순간 Application 이 Domain 을 침식한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import InspectionView
from application.data.support import commit, load_dataset
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository
from domain.data.schema import DataSchema


@dataclass(frozen=True, slots=True)
class DeclareDataSchemaCommand:
    dataset_id: str
    schema: DataSchema


class DeclareDataSchema:
    def __init__(
        self,
        repository: DatasetRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: DeclareDataSchemaCommand) -> InspectionView:
        dataset = load_dataset(self._repository, command.dataset_id)
        report = dataset.declare_schema(command.schema)
        commit(self._repository, dataset, self._publisher)
        return InspectionView.of(str(dataset.id), report)
