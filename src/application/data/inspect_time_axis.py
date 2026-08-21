"""InspectTimeAxis — 시간축을 검증한다. (실습 1-5)"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import InspectionView
from application.data.support import commit, load_dataset
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository, TimeAxisMeasurer
from domain.data.time_axis import TimeAxisPolicy


@dataclass(frozen=True, slots=True)
class InspectTimeAxisCommand:
    dataset_id: str
    policy: TimeAxisPolicy


class InspectTimeAxis:
    def __init__(
        self,
        repository: DatasetRepository,
        measurer: TimeAxisMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._measurer = measurer
        self._publisher = publisher

    def execute(self, command: InspectTimeAxisCommand) -> InspectionView:
        dataset = load_dataset(self._repository, command.dataset_id)
        schema = dataset.schema
        if schema is None:
            raise UnsupportedOperation(
                "스키마가 없다.", subject=str(dataset.id)
            )
        time_index = schema.time_index
        if time_index is None:
            raise UnsupportedOperation(
                "시간축(TIME_INDEX)이 선언되지 않았다. 시계열이 아니거나, 선언을 빠뜨렸다.",
                subject=str(dataset.id),
            )

        measurement = self._measurer.measure(dataset.source, time_index.name)
        report = command.policy.inspect(measurement)
        dataset.record_inspection(report)
        commit(self._repository, dataset, self._publisher)
        return InspectionView.of(str(dataset.id), report)
