"""ReduceStructure — 구조를 줄이는 것과 숫자를 줄이는 것은 다르다. (실습 4-11)

여러 축소 방법을 실제로 적용해 보고, 각각 무엇이 줄었는지 Domain 이 판정한다.

**"파라미터가 절반이 되었다"와 "빨라졌다"는 다른 말이다.**
이 Use Case 는 그 둘을 따로 세어서 나란히 놓는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.optimization.dto import ReductionComparisonView
from application.optimization.support import load_run
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.optimization.ports import OptimizationRunRepository, StructuralReducer
from domain.optimization.structural import (
    ReductionComparison,
    StructuralPolicy,
    StructuralReduction,
)


@dataclass(frozen=True, slots=True)
class ReduceStructureCommand:
    run_id: str
    reductions: tuple[tuple[str, StructuralReduction], ...]
    """(이름, 축소 방법). 이름은 비교표에 그대로 나온다."""

    split: str = "test"
    fine_tune_epochs: int = 6
    policy: StructuralPolicy = field(default_factory=StructuralPolicy)


class ReduceStructure:
    def __init__(
        self,
        runs: OptimizationRunRepository,
        reducer: StructuralReducer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._reducer = reducer
        self._publisher = publisher

    def execute(self, command: ReduceStructureCommand) -> ReductionComparisonView:
        if not command.reductions:
            raise UnsupportedOperation(
                "비교할 축소 방법이 없다.", subject=command.run_id
            )

        run = load_run(self._runs, command.run_id)
        baseline = run.baseline

        rows = []
        for label, reduction in command.reductions:
            outcome = self._reducer.reduce(
                baseline,
                reduction,
                label=label,
                split=command.split,
                fine_tune_epochs=command.fine_tune_epochs,
            )
            rows.append((outcome, command.policy.inspect(outcome)))

        comparison = ReductionComparison(rows=tuple(rows))
        return ReductionComparisonView.of(
            command.run_id,
            comparison,
            findings=tuple(
                FindingView.of(f) for _, findings in rows for f in findings
            ),
        )
