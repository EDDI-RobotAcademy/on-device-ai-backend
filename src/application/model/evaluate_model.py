"""EvaluateModel — Accuracy 뒤에 숨어 있는 실패를 찾아라. (실습 3-9)

정확도 하나만 돌려주지 않는다. 혼동 행렬과 클래스별 재현율을 함께 돌려준다.
그리고 지연시간도 같이 잰다 — 정확도만 재는 평가는 절반짜리다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from application.data.dto import FindingView
from application.model.dto import EvaluationView
from application.model.support import commit, load_run
from application.shared.ports import EventPublisher
from domain.model.evaluation import EvaluationPolicy
from domain.model.ports import (
    FieldEvaluator,
    ModelEvaluator,
    TrainingRunRepository,
)


@dataclass(frozen=True, slots=True)
class EvaluateModelCommand:
    run_id: str
    split: str = "test"
    policy: EvaluationPolicy = field(default_factory=EvaluationPolicy)


class EvaluateModel:
    def __init__(
        self,
        runs: TrainingRunRepository,
        evaluator: ModelEvaluator,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._evaluator = evaluator
        self._publisher = publisher

    def execute(self, command: EvaluateModelCommand) -> EvaluationView:
        run = load_run(self._runs, command.run_id)
        if run.model_version_id is None:
            from domain.model.errors import ModelNotTrained

            raise ModelNotTrained(
                "학습이 끝나지 않아 평가할 모델이 없다.", subject=str(run.id)
            )

        result = self._evaluator.evaluate(run.model_version_id, command.split)
        run.record_evaluation(result)
        commit(self._runs, run, self._publisher)

        findings = command.policy.inspect(result)
        return EvaluationView.of(
            str(run.id), result, tuple(FindingView.of(f) for f in findings)
        )


@dataclass(frozen=True, slots=True)
class EvaluateOnFieldCommand:
    """현장 홀드아웃으로 평가한다. (실습 3-10)"""

    run_id: str
    field_uri: str
    split_name: str = "field"
    policy: EvaluationPolicy = field(default_factory=EvaluationPolicy)


class EvaluateModelOnField:
    """다른 날 데이터로 평가한다.

    정규화 통계는 **학습 때 쓰던 것을 그대로** 쓴다 (실습 1-7).
    현장 데이터로 통계를 다시 뽑으면 그건 다른 전처리이고, 배포된 모델과 다르게 동작한다.
    """

    def __init__(
        self,
        runs: TrainingRunRepository,
        evaluator: FieldEvaluator,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._evaluator = evaluator
        self._publisher = publisher

    def execute(self, command: EvaluateOnFieldCommand) -> EvaluationView:
        run = load_run(self._runs, command.run_id)
        if run.model_version_id is None:
            from domain.model.errors import ModelNotTrained

            raise ModelNotTrained(
                "학습이 끝나지 않아 평가할 모델이 없다.", subject=str(run.id)
            )

        field_data = replace(
            run.data,
            dataset_ref=f"{run.data.dataset_ref}/field",
            uri=command.field_uri,
        )
        result = self._evaluator.evaluate_external(
            run.model_version_id,
            field_data,
            window_length=run.windowing.window_length,
            stride=run.windowing.stride,
            label_policy=run.windowing.label_policy,
            split_name=command.split_name,
        )
        run.record_evaluation(result)
        commit(self._runs, run, self._publisher)

        findings = command.policy.inspect(result)
        return EvaluationView.of(
            str(run.id), result, tuple(FindingView.of(f) for f in findings)
        )
