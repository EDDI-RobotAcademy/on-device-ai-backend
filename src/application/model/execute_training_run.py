"""ExecuteTrainingRun — 첫 번째 모델을 직접 학습시켜라. (실습 3-5 ~ 3-8)

Job 을 실제로 돌리는 Use Case 다 (CLAUDE.md §11).
HTTP 라우트는 이것을 백그라운드로 던지고 즉시 응답한다.

Application 이 하는 일은 네 줄이다.
    run.start()                 상태를 RUNNING 으로
    trainer.train(...)          Infrastructure 가 계산한다
    run.record_epoch(...)       결과를 Aggregate 에 기록
    run.complete(version_id)    끝났다고 표시

학습률을 조정하거나, 손실이 이상하면 멈추는 판단은 여기에 없다.
그건 Domain(LearningPolicy / OverfittingPolicy)의 일이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import FindingView
from application.model.dto import TrainingCurveView
from application.model.support import commit, load_run
from application.shared.ports import EventPublisher
from domain.model.curve import LearningPolicy, OverfittingPolicy
from domain.model.identifiers import ModelVersionId
from domain.model.ports import ModelTrainer, TrainingRunRepository
from domain.model.tensor_spec import BatchSpec


@dataclass(frozen=True, slots=True)
class ExecuteTrainingRunCommand:
    run_id: str
    learning_policy: LearningPolicy = LearningPolicy()
    overfitting_policy: OverfittingPolicy = OverfittingPolicy()


class ExecuteTrainingRun:
    def __init__(
        self,
        runs: TrainingRunRepository,
        trainer: ModelTrainer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._trainer = trainer
        self._publisher = publisher

    def execute(self, command: ExecuteTrainingRunCommand) -> TrainingCurveView:
        run = load_run(self._runs, command.run_id)
        run.start()
        commit(self._runs, run, self._publisher)

        batch = BatchSpec(
            sample=run.architecture.input_spec, batch_size=run.config.batch_size
        )
        try:
            outcome = self._trainer.train(
                run.data,
                run.architecture,
                run.config,
                batch,
                run.windowing.window_length,
                run.windowing.stride,
                run.windowing.label_policy,
            )
        except Exception as exc:  # noqa: BLE001 - 실패도 기록되어야 한다
            run.fail(f"{type(exc).__name__}: {exc}")
            commit(self._runs, run, self._publisher)
            raise

        for record in outcome.epochs:
            run.record_epoch(record)

        run.attach_split_usage(outcome.usage)
        run.complete(ModelVersionId.of(outcome.artifact_uri.rsplit("/", 1)[-1]))
        commit(self._runs, run, self._publisher)

        baseline = _baseline_accuracy(run)
        findings = command.learning_policy.inspect(
            run.curve, baseline
        ) + command.overfitting_policy.inspect(run.curve)

        return TrainingCurveView.of(
            str(run.id),
            run.status.value,
            run.curve,
            tuple(FindingView.of(f) for f in findings),
        )


def _baseline_accuracy(run) -> float:  # noqa: ANN001
    """검증 분할에서 다수 클래스만 찍었을 때의 정확도."""
    summary = run.tensor_summaries.get("validation")
    if summary is None or not summary.class_counts:
        return 0.0
    total = sum(summary.class_counts.values())
    if total == 0:
        return 0.0
    return max(summary.class_counts.values()) / total
