"""SelectModel — 가장 빠른 모델이 아니라 가장 쓸 수 있는 모델. (실습 4-10)

선택 규칙 자체는 Domain(SelectionPolicy)에 있다.
이 Use Case 가 하는 일은 예산을 정책으로 감싸 Aggregate 에 건네는 것뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.optimization.dto import SelectionView
from application.optimization.support import commit, load_run
from application.shared.ports import EventPublisher
from domain.optimization.conversion import EquivalencePolicy
from domain.optimization.ports import OptimizationRunRepository
from domain.optimization.selection import (
    DeviceBudget,
    SelectionObjective,
    SelectionPolicy,
)


@dataclass(frozen=True, slots=True)
class SelectModelCommand:
    run_id: str
    budget: DeviceBudget
    objective: SelectionObjective = SelectionObjective.ACCURACY
    equivalence: EquivalencePolicy = field(default_factory=EquivalencePolicy)
    require_deployable_runtime: bool = True


class SelectModel:
    def __init__(
        self,
        runs: OptimizationRunRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._publisher = publisher

    def execute(self, command: SelectModelCommand) -> SelectionView:
        run = load_run(self._runs, command.run_id)
        policy = SelectionPolicy(
            budget=command.budget,
            objective=command.objective,
            equivalence=command.equivalence,
            require_deployable_runtime=command.require_deployable_runtime,
        )
        certificate = run.select(policy)
        commit(self._runs, run, self._publisher)
        return SelectionView.of(str(run.id), certificate)


@dataclass(frozen=True, slots=True)
class ReopenOptimizationRunCommand:
    run_id: str
    reason: str


class ReopenOptimizationRun:
    """판정을 되돌린다. **이유 없이는 되돌릴 수 없다.**"""

    def __init__(
        self,
        runs: OptimizationRunRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._publisher = publisher

    def execute(self, command: ReopenOptimizationRunCommand) -> str:
        run = load_run(self._runs, command.run_id)
        run.reopen(command.reason)
        commit(self._runs, run, self._publisher)
        return run.status.value
