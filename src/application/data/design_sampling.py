"""DesignSampling — 수집 주기와 해상도를 직접 정하라. (실습 1-11)

여러 주기를 실제로 뽑아 보고, 각각에 대해 Domain 이 판정한다.
그 결과를 한 표로 돌려준다.

**고르는 것은 사람이다.** 이 Use Case 는 "통과하는 것 중 가장 싼 것"까지만 말한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView, SamplingTradeoffView
from application.data.support import load_dataset
from application.shared.errors import UnsupportedOperation
from domain.data.ports import DatasetRepository, SamplingProbe
from domain.data.sampling_design import (
    SamplingDesignPolicy,
    SamplingPlan,
    SamplingTradeoff,
)


@dataclass(frozen=True, slots=True)
class DesignSamplingCommand:
    dataset_id: str
    plans: tuple[SamplingPlan, ...]
    normal_label: str = "NORMAL"
    value_field: str | None = None
    policy: SamplingDesignPolicy = field(default_factory=SamplingDesignPolicy)


class DesignSampling:
    def __init__(
        self, datasets: DatasetRepository, probe: SamplingProbe
    ) -> None:
        self._datasets = datasets
        self._probe = probe

    def execute(self, command: DesignSamplingCommand) -> SamplingTradeoffView:
        if not command.plans:
            raise UnsupportedOperation(
                "비교할 설계가 없다. 최소 둘은 있어야 고를 수 있다.",
                subject=command.dataset_id,
            )

        dataset = load_dataset(self._datasets, command.dataset_id)
        schema = dataset.schema
        if schema is None or schema.time_index is None:
            raise UnsupportedOperation(
                "시간축이 확정되지 않은 Dataset 으로는 수집 주기를 설계할 수 없다. "
                "실습 1-5 를 먼저 통과해야 한다.",
                subject=command.dataset_id,
            )
        label_field = _label_field(dataset)

        rows = []
        for plan in command.plans:
            observation = self._probe.probe(
                dataset.source.uri,
                dataset.source.format.value,
                time_field=schema.time_index.name,
                label_field=label_field,
                normal_label=command.normal_label,
                plan=plan,
                value_field=command.value_field,
            )
            rows.append((plan, observation, command.policy.inspect(plan, observation)))

        tradeoff = SamplingTradeoff(rows=tuple(rows))
        return SamplingTradeoffView.of(
            str(dataset.id),
            tradeoff,
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
        "라벨 열이 정해지지 않았다. 무엇을 '사건'으로 볼지 모르면 "
        "주기를 정할 수 없다 (실습 1-6).",
        subject=str(dataset.id),
    )
