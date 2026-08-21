"""ExecuteImageTrainingRun — 이미지 모델을 실제로 학습시켜라. (실습 3-11)

`ExecuteTrainingRun` 과 하는 일이 같다.

    run.start() → trainer.train(...) → run.record_epoch(...) → run.complete(...)

Loss 가 안 떨어지는지, 외우기 시작했는지 판단하는 것도 같은 Domain Policy 다
(LearningPolicy / OverfittingPolicy). **판단 기준은 재료가 달라도 같다.**
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import FindingView
from application.model.dto import TrainingCurveView
from application.model.support import commit, load_run
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.model.curve import LearningPolicy, OverfittingPolicy
from domain.model.identifiers import ModelVersionId
from domain.model.ports import ImageModelTrainer, TrainingRunRepository
from domain.model.tensor_spec import BatchSpec


@dataclass(frozen=True, slots=True)
class ExecuteImageTrainingRunCommand:
    run_id: str
    learning_policy: LearningPolicy = LearningPolicy()
    overfitting_policy: OverfittingPolicy = OverfittingPolicy()


class ExecuteImageTrainingRun:
    def __init__(
        self,
        runs: TrainingRunRepository,
        trainer: ImageModelTrainer,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._trainer = trainer
        self._publisher = publisher

    def execute(self, command: ExecuteImageTrainingRunCommand) -> TrainingCurveView:
        run = load_run(self._runs, command.run_id)
        if not run.is_image_run:
            raise UnsupportedOperation(
                "이 학습은 이미지 학습이 아니다. ExecuteTrainingRun 을 써야 한다.",
                subject=command.run_id,
            )

        run.start()
        commit(self._runs, run, self._publisher)

        batch = BatchSpec(
            sample=run.architecture.input_spec, batch_size=run.config.batch_size
        )
        try:
            outcome = self._trainer.train(
                run.image_data, run.architecture, run.config, batch
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

        findings = command.learning_policy.inspect(
            run.curve, _baseline_accuracy(run)
        ) + command.overfitting_policy.inspect(run.curve)

        return TrainingCurveView.of(
            str(run.id),
            run.status.value,
            run.curve,
            tuple(FindingView.of(f) for f in findings),
        )


def _baseline_accuracy(run) -> float:  # noqa: ANN001
    """가장 많은 폴더만 찍었을 때의 정확도.

    이미지에서는 이 숫자를 잊기 쉽다. 폴더가 2개면 대개 0.5 근처지만,
    OK 가 NG 보다 훨씬 많은 현장 데이터에서는 0.9 를 넘기도 한다.
    """
    summary = run.tensor_summaries.get("all")
    if summary is None or not summary.class_counts:
        return 0.0
    total = sum(summary.class_counts.values())
    return max(summary.class_counts.values()) / total if total else 0.0
