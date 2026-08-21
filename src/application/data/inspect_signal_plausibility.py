"""InspectSignalPlausibility — 센서/이미지가 물리적으로 말이 되는지 본다. (실습 1-4)

Modality 에 따라 다른 측정기를 쓴다. 그 분기 자체는 기술 선택이 아니라
"센서와 이미지는 다르게 거짓말한다"는 도메인 사실의 반영이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import InspectionView
from application.data.support import commit, load_dataset
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository, ImageSignalMeasurer, SensorSignalMeasurer
from domain.data.signal import SignalPlausibilityPolicy
from domain.data.source import Modality


@dataclass(frozen=True, slots=True)
class InspectSignalPlausibilityCommand:
    dataset_id: str
    policy: SignalPlausibilityPolicy = field(default_factory=SignalPlausibilityPolicy)


class InspectSignalPlausibility:
    def __init__(
        self,
        repository: DatasetRepository,
        sensor_measurer: SensorSignalMeasurer,
        image_measurer: ImageSignalMeasurer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._sensor_measurer = sensor_measurer
        self._image_measurer = image_measurer
        self._publisher = publisher

    def execute(self, command: InspectSignalPlausibilityCommand) -> InspectionView:
        dataset = load_dataset(self._repository, command.dataset_id)
        schema = dataset.schema
        if schema is None:
            raise UnsupportedOperation(
                "스키마가 없다. 어느 열이 어떤 물리량인지 모르면 신호를 검증할 수 없다.",
                subject=str(dataset.id),
            )

        modality = dataset.source.modality
        if modality is Modality.IMAGE:
            measurement = self._image_measurer.measure(dataset.source)
            report = command.policy.inspect_images(measurement)
        elif modality in (Modality.TIME_SERIES, Modality.TABULAR):
            measurements = self._sensor_measurer.measure(dataset.source, schema)
            report = command.policy.inspect_sensors(measurements)
        else:  # pragma: no cover - Modality 가 늘어나면 여기서 걸린다
            raise UnsupportedOperation(
                f"{modality.value} 는 아직 신호 검증을 지원하지 않는다.",
                subject=modality.value,
            )

        dataset.record_inspection(report)
        commit(self._repository, dataset, self._publisher)
        return InspectionView.of(str(dataset.id), report)
