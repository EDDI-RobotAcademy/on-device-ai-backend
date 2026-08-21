"""AnalyzeRepresentativeness — 학습 데이터가 현실을 대표하는지 본다. (실습 1-9)

reference = 학습에 쓸 데이터, observed = 최근 현장에서 받은 표본.
같은 계산을 배포 후에 반복하면 그것이 Data Drift 감시가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import InspectionView
from application.data.support import commit, load_dataset
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository, DistributionComparer
from domain.data.representativeness import RepresentativenessPolicy
from domain.data.source import DataSourceDescriptor


@dataclass(frozen=True, slots=True)
class AnalyzeRepresentativenessCommand:
    dataset_id: str
    observed: DataSourceDescriptor
    policy: RepresentativenessPolicy = field(default_factory=RepresentativenessPolicy)


@dataclass(frozen=True, slots=True)
class RepresentativenessView:
    dataset_id: str
    observed_uri: str
    worst_field: str | None
    worst_psi: float
    field_psi: tuple[tuple[str, float, float], ...]
    """(필드명, PSI, coverage_ratio)."""

    inspection: InspectionView

    def render(self) -> str:
        lines = [
            f"대표성 비교 ({self.dataset_id}) ← {self.observed_uri}",
            f"{'field':<20}{'PSI':>10}{'coverage':>12}",
            "-" * 42,
        ]
        for name, psi, coverage in self.field_psi:
            flag = "  ← 심각" if psi >= 0.25 else ("  ← 이동중" if psi >= 0.10 else "")
            lines.append(f"{name:<20}{psi:>10.4f}{coverage:>11.1%}{flag}")
        lines.append(self.inspection.render())
        return "\n".join(lines)


class AnalyzeRepresentativeness:
    def __init__(
        self,
        repository: DatasetRepository,
        comparer: DistributionComparer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._comparer = comparer
        self._publisher = publisher

    def execute(self, command: AnalyzeRepresentativenessCommand) -> RepresentativenessView:
        dataset = load_dataset(self._repository, command.dataset_id)
        schema = dataset.schema
        if schema is None:
            raise UnsupportedOperation("스키마가 없다.", subject=str(dataset.id))

        measurement = self._comparer.compare(dataset.source, command.observed, schema)
        report = command.policy.inspect(measurement)
        dataset.record_inspection(report)
        commit(self._repository, dataset, self._publisher)

        return RepresentativenessView(
            dataset_id=str(dataset.id),
            observed_uri=command.observed.uri,
            worst_field=measurement.worst_field,
            worst_psi=measurement.worst_psi,
            field_psi=tuple(
                (s.field_name, s.psi, s.coverage_ratio) for s in measurement.field_shifts
            ),
            inspection=InspectionView.of(str(dataset.id), report),
        )
