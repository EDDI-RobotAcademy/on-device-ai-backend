"""StartOptimizationRun — 최적화의 출발점을 고정한다. (실습 4-1)

최적화는 "빠르게 만들기"가 아니라 **비교**다.
비교하려면 기준이 있어야 하고, 기준은 승인받은 모델이어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.model.support import load_run as load_training_run
from application.optimization.baseline_mapper import baseline_from
from application.optimization.dto import OptimizationRunView
from application.optimization.support import commit
from application.shared.ports import EventPublisher
from domain.model.ports import TrainingRunRepository
from domain.optimization.identifiers import OptimizationRunId
from domain.optimization.optimization_run import OptimizationRun
from domain.optimization.ports import OptimizationRunRepository


@dataclass(frozen=True, slots=True)
class StartOptimizationRunCommand:
    run_id: str
    training_run_id: str
    split: str = "test"
    require_accepted: bool = True
    """승인 게이트를 끄고 시작할 수도 있다. **끄는 것은 기록으로 남는 선택이다.**"""


class StartOptimizationRun:
    """모듈 3 의 학습 결과를 최적화의 기준으로 번역해 들여온다.

    번역은 `baseline_from` 이 하고, 통과 여부는 Domain 이 판단한다.
    Use Case 는 둘을 잇기만 한다.
    """

    def __init__(
        self,
        runs: OptimizationRunRepository,
        training_runs: TrainingRunRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._training_runs = training_runs
        self._publisher = publisher

    def execute(self, command: StartOptimizationRunCommand) -> OptimizationRunView:
        training_run = load_training_run(self._training_runs, command.training_run_id)
        baseline = baseline_from(training_run, split=command.split)

        run = OptimizationRun.start(
            OptimizationRunId.of(command.run_id),
            baseline,
            require_accepted=command.require_accepted,
        )
        commit(self._runs, run, self._publisher)
        return OptimizationRunView.of(run)
