"""DefineLabelSpace — 정상과 이상의 정의를 확정하고 실제 라벨과 대조한다. (실습 1-6)"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import InspectionView
from application.data.support import commit, load_dataset
from application.shared.ports import EventPublisher
from domain.data.labeling import LabelPolicy, LabelSpace
from domain.data.ports import DatasetRepository, LabelMeasurer


@dataclass(frozen=True, slots=True)
class DefineLabelSpaceCommand:
    dataset_id: str
    label_space: LabelSpace
    policy: LabelPolicy = field(default_factory=LabelPolicy)


class DefineLabelSpace:
    def __init__(
        self,
        repository: DatasetRepository,
        measurer: LabelMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._measurer = measurer
        self._publisher = publisher

    def execute(self, command: DefineLabelSpaceCommand) -> InspectionView:
        dataset = load_dataset(self._repository, command.dataset_id)

        # 정의를 먼저 확정한다. 정의 없이 측정하면 무엇과 비교할지 알 수 없다.
        dataset.define_label_space(command.label_space)

        measurement = self._measurer.measure(
            dataset.source, command.label_space.field_name
        )
        report = command.policy.inspect(command.label_space, measurement)
        dataset.record_inspection(report)
        commit(self._repository, dataset, self._publisher)
        return InspectionView.of(str(dataset.id), report)
