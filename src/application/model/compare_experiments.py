"""CompareExperiments — 실험을 나란히 놓고 비교하라. (실습 3-12, 3-14, 3-15)

이미 끝난 TrainingRun 들을 읽어 비교표를 만든다. **새로 학습하지 않는다.**

Application 이 하는 일은 셋이다.
    1. 각 학습에서 숫자를 꺼낸다 (Domain 객체 → Trial)
    2. 무엇을 다르게 두었는지(손잡이)를 함께 붙인다
    3. Domain Policy 에게 "이 비교를 믿어도 되는가"를 묻는다

무엇이 더 나은가를 여기서 정하지 않는다. 그건 ExperimentBoard 의 일이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.model.dto import ExperimentBoardView
from application.model.support import load_run
from application.shared.errors import UnsupportedOperation
from domain.model.experiment import (
    ExperimentBoard,
    ExperimentPolicy,
    ExperimentTrial,
    TrialKnobs,
    TrialMetrics,
)
from domain.model.ports import TrainingRunRepository
from domain.model.training_run import TrainingRun


@dataclass(frozen=True, slots=True)
class TrialRequest:
    """비교에 넣을 학습 하나."""

    run_id: str
    label: str
    knobs: Mapping[str, str]
    split: str = "test"


@dataclass(frozen=True, slots=True)
class CompareExperimentsCommand:
    name: str
    trials: tuple[TrialRequest, ...]
    metric: str = "macro_f1"
    policy: ExperimentPolicy = field(default_factory=ExperimentPolicy)


class CompareExperiments:
    def __init__(self, runs: TrainingRunRepository) -> None:
        self._runs = runs

    def execute(self, command: CompareExperimentsCommand) -> ExperimentBoardView:
        board = ExperimentBoard(name=command.name)
        for request in command.trials:
            run = load_run(self._runs, request.run_id)
            board = board.with_trial(_trial_of(run, request))

        findings = command.policy.inspect(board, metric=command.metric)
        return ExperimentBoardView.of(
            board,
            metric=command.metric,
            findings=tuple(FindingView.of(f) for f in findings),
        )


def _trial_of(run: TrainingRun, request: TrialRequest) -> ExperimentTrial:
    result = run.evaluation_of(request.split)
    if result is None:
        raise UnsupportedOperation(
            f"'{request.run_id}' 에 '{request.split}' 평가가 없다. "
            "평가하지 않은 학습은 비교에 넣을 수 없다.",
            subject=request.run_id,
        )
    profile = run.profile
    return ExperimentTrial(
        label=request.label,
        knobs=TrialKnobs(values=dict(request.knobs)),
        metrics=TrialMetrics(
            accuracy=result.accuracy,
            macro_recall=result.macro_recall,
            macro_f1=result.matrix.macro_f1,
            loss=result.loss,
            latency_ms_p50=result.latency_ms_p50,
            parameter_count=profile.parameter_count if profile else 0,
            epochs=len(run.curve),
            evaluated_samples=result.matrix.total,
        ),
        seed=run.config.seed,
        data_ref=run.data.dataset_ref,
    )
