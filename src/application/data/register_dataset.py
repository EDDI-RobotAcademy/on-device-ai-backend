"""RegisterDataset — 현장 데이터를 시스템이 아는 대상으로 만든다. (실습 1-1)

여기서 파일을 읽지 않는다. "이런 데이터가 어디에 있고 어느 현장에서 왔다"까지만 등록한다.
읽는 것은 다음 Use Case(ProfileDataset)의 일이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import DatasetView
from application.data.support import commit
from application.shared.errors import ConflictingRequest
from application.shared.ports import EventPublisher
from domain.data.dataset import Dataset
from domain.data.identifiers import DatasetId
from domain.data.ports import DatasetRepository
from domain.data.source import DataSourceDescriptor, Modality, SourceFormat


@dataclass(frozen=True, slots=True)
class RegisterDatasetCommand:
    dataset_id: str
    name: str
    uri: str
    source_format: SourceFormat
    modality: Modality
    collected_from: str


class RegisterDataset:
    def __init__(
        self,
        repository: DatasetRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: RegisterDatasetCommand) -> DatasetView:
        dataset_id = DatasetId.of(command.dataset_id)
        if self._repository.exists(dataset_id):
            raise ConflictingRequest(
                f"Dataset '{dataset_id}' 은 이미 등록되어 있다.", subject=str(dataset_id)
            )

        source = DataSourceDescriptor(
            uri=command.uri,
            format=command.source_format,
            modality=command.modality,
            collected_from=command.collected_from,
        )
        dataset = Dataset.register(dataset_id, command.name, source)
        commit(self._repository, dataset, self._publisher)
        return DatasetView.of(dataset)
