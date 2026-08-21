"""CompareRebalancing — 불균형을 줄이는 방법마다 잃는 것이 다르다. (실습 2-11)

여러 전략을 실제로 적용해 보고, 각각 무엇을 잃었는지 Domain 이 판정한다.

**순위를 매기지 않는다.** 이 Use Case 는 대가를 나란히 놓을 뿐이다.
무엇을 감수할지는 현장이 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.data.support import load_dataset
from application.data_quality.dto import RebalancingComparisonView
from application.shared.errors import UnsupportedOperation
from domain.data.ports import DatasetRepository
from domain.data_quality.ports import Resampler
from domain.data_quality.rebalancing import (
    RebalancingComparison,
    RebalancingPlan,
    RebalancingPolicy,
)


@dataclass(frozen=True, slots=True)
class CompareRebalancingCommand:
    dataset_id: str
    plans: tuple[RebalancingPlan, ...]
    policy: RebalancingPolicy = field(default_factory=RebalancingPolicy)


class CompareRebalancing:
    def __init__(self, datasets: DatasetRepository, resampler: Resampler) -> None:
        self._datasets = datasets
        self._resampler = resampler

    def execute(
        self, command: CompareRebalancingCommand
    ) -> RebalancingComparisonView:
        if not command.plans:
            raise UnsupportedOperation(
                "비교할 전략이 없다. **아무것도 하지 않는 것**도 하나의 전략이므로 "
                "최소한 그것과는 비교해야 한다.",
                subject=command.dataset_id,
            )

        dataset = load_dataset(self._datasets, command.dataset_id)
        label_field = _label_field(dataset)

        rows = []
        for plan in command.plans:
            outcome = self._resampler.resample(
                dataset.source.uri,
                dataset.source.format.value,
                label_field=label_field,
                plan=plan,
            )
            rows.append((plan, outcome, command.policy.inspect(plan, outcome)))

        comparison = RebalancingComparison(rows=tuple(rows))
        return RebalancingComparisonView.of(
            str(dataset.id),
            comparison,
            findings=tuple(
                FindingView.of(f) for _, _, findings in rows for f in findings
            ),
        )


def _label_field(dataset) -> str:  # noqa: ANN001
    space = dataset.label_space
    if space is not None:
        return space.field_name
    spec = dataset.training_spec
    if spec is not None and spec.label_field:
        return spec.label_field
    raise UnsupportedOperation(
        "라벨 열이 정해지지 않았다. 무엇의 균형을 맞출지 모른다 (실습 1-6).",
        subject=str(dataset.id),
    )
