"""PartitionDataset — 계획대로 나누고 누수를 확인한다. (실습 1-8)"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import InspectionView
from application.data.support import commit, load_dataset
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data.partition import PartitionPlan, PartitionPolicy
from domain.data.ports import DatasetRepository, PartitionEngine


@dataclass(frozen=True, slots=True)
class PartitionDatasetCommand:
    dataset_id: str
    plan: PartitionPlan
    policy: PartitionPolicy = field(default_factory=PartitionPolicy)


@dataclass(frozen=True, slots=True)
class PartitionView:
    dataset_id: str
    strategy: str
    train_count: int
    validation_count: int
    test_count: int
    overlapping_group_count: int
    time_overlap_seconds: float
    inspection: InspectionView

    def render(self) -> str:
        total = self.train_count + self.validation_count + self.test_count
        lines = [
            f"분할 결과 ({self.dataset_id}) — 전략 {self.strategy}",
            f"  train      : {self.train_count:>8,}  ({self.train_count / total:.1%})"
            if total
            else "  train      : 0",
            f"  validation : {self.validation_count:>8,}",
            f"  test       : {self.test_count:>8,}",
            f"  그룹 누수  : {self.overlapping_group_count} 개 그룹",
            f"  시간 누수  : {self.time_overlap_seconds:g} 초",
            self.inspection.render(),
        ]
        return "\n".join(lines)


class PartitionDataset:
    def __init__(
        self,
        repository: DatasetRepository,
        engine: PartitionEngine,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._engine = engine
        self._publisher = publisher

    def execute(self, command: PartitionDatasetCommand) -> PartitionView:
        dataset = load_dataset(self._repository, command.dataset_id)
        schema = dataset.schema
        if schema is None:
            raise UnsupportedOperation("스키마가 없다.", subject=str(dataset.id))

        # 계획이 이 스키마에서 성립하는지 먼저 따진다.
        # 시계열을 RANDOM 으로 나누려는 시도는 여기서 InvariantViolation 으로 막힌다.
        command.plan.validate_against(schema)

        label_field = dataset.label_space.field_name if dataset.label_space else None
        measurement = self._engine.apply(
            dataset.source, schema, command.plan, label_field
        )
        report = command.policy.inspect(command.plan, measurement)
        dataset.apply_partition(command.plan, measurement, report)
        commit(self._repository, dataset, self._publisher)

        return PartitionView(
            dataset_id=str(dataset.id),
            strategy=command.plan.strategy.value,
            train_count=measurement.train_count,
            validation_count=measurement.validation_count,
            test_count=measurement.test_count,
            overlapping_group_count=measurement.overlapping_group_count,
            time_overlap_seconds=measurement.time_overlap_seconds,
            inspection=InspectionView.of(str(dataset.id), report),
        )
