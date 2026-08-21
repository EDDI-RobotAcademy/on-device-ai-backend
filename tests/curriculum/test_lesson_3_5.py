"""실습 3-5 — 첫 번째 모델을 직접 학습시켜라.

    pytest -m lesson_3_5 -s

학습은 오래 걸리는 작업이다. HTTP 요청이 그것을 붙잡고 기다리는 구조를 만들지 않는다.

    Command  →  Job          →  Job Status         →  Result
    학습 요청 →  TrainingRun  →  RUNNING/COMPLETED  →  ModelVersion
"""

from __future__ import annotations

import pytest

from application.model.get_training_run import GetTrainingRunQuery
from domain.model.curve import EpochRecord
from domain.model.training_config import EarlyStoppingRule, TrainingConfig
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from tests.support import model_scenario as ms
from tests.support import report

pytestmark = pytest.mark.lesson_3_5


def test_학습은_Job_이다(trained) -> None:
    report.section("실습 3-5 · 첫 번째 모델을 직접 학습시켜라")

    view = trained.model.get_training_run().execute(
        GetTrainingRunQuery(run_id=trained.run_id)
    )
    report.block(
        "학습 기록",
        f"  상태        : {view.status}\n"
        f"  데이터      : {view.dataset_ref}\n"
        f"  구조        : {view.architecture}\n"
        f"  설정        : {view.config}\n"
        f"  창          : {view.windowing}\n"
        f"  epoch       : {view.epoch_count} (최저점 {view.best_epoch})\n"
        f"  모델 버전   : {view.model_version_id}",
    )

    assert view.status == "COMPLETED"
    assert view.epoch_count == 10
    assert view.model_version_id is not None
    report.note(
        "학습이 끝나야 ModelVersion 이 생긴다. "
        "그 전에는 '학습 중'이라는 상태만 있다."
    )


def test_설정_하나하나가_결과를_바꾼다(trained) -> None:
    """그래서 전부 기록되어야 한다. 특히 seed."""
    from application.model.support import load_run

    run = load_run(trained.model.runs, trained.run_id)
    config = run.config

    assert config.seed == 42
    assert "seed=42" in config.describe()
    report.note(
        f"설정: {config.describe()}\n"
        "     seed 가 없으면 '어제는 됐는데요'를 설명할 방법이 없다."
    )


def test_말이_안_되는_설정은_객체가_거부한다() -> None:
    with pytest.raises(InvariantViolation, match="learning_rate"):
        TrainingConfig(learning_rate=5.0)
    with pytest.raises(InvariantViolation, match="epochs"):
        TrainingConfig(epochs=0)
    with pytest.raises(InvariantViolation, match="patience"):
        EarlyStoppingRule(patience=0)

    report.note("학습을 3분 돌린 뒤가 아니라 설정을 만드는 순간에 걸린다.")


def test_상태_전이는_순서를_지킨다(trained) -> None:
    """이미 끝난 학습을 다시 시작할 수 없다."""
    with pytest.raises(IllegalStateTransition, match="학습을 시작할 수 없다"):
        ms.execute(trained.model, trained.run_id)


def test_epoch_은_순서대로만_기록된다() -> None:
    """중간에 하나를 빠뜨리거나 뒤섞으면 곡선을 읽을 수 없다."""
    from domain.model.identifiers import TrainingRunId
    from domain.model.training_data_ref import TrainingDataRef
    from domain.model.training_run import TrainingRun

    data = TrainingDataRef(
        dataset_ref="ds",
        uri="x.csv",
        feature_fields=tuple(f"f{i}" for i in range(6)),
        label_field="condition",
        readiness_certified=True,
        quality_gate_passed=True,
    )
    run = TrainingRun.prepare(
        TrainingRunId.of("r"),
        data,
        ms.cnn_architecture(),
        ms.training_config(),
        ms.windowing_plan(),
    )
    run.start()
    run.record_epoch(EpochRecord(1, 1.0, 1.0, 0.5, 0.5))

    with pytest.raises(InvariantViolation, match="순서대로"):
        run.record_epoch(EpochRecord(1, 0.9, 0.9, 0.6, 0.6))


def test_실패도_기록된다() -> None:
    """왜 실패했는지 없으면 다시 시도할 근거가 없다."""
    from domain.model.identifiers import TrainingRunId
    from domain.model.training_data_ref import TrainingDataRef
    from domain.model.training_run import TrainingRun, TrainingStatus

    data = TrainingDataRef(
        dataset_ref="ds",
        uri="x.csv",
        feature_fields=tuple(f"f{i}" for i in range(6)),
        label_field="condition",
        readiness_certified=True,
        quality_gate_passed=True,
    )
    run = TrainingRun.prepare(
        TrainingRunId.of("r"),
        data,
        ms.cnn_architecture(),
        ms.training_config(),
        ms.windowing_plan(),
    )
    run.start()

    with pytest.raises(InvariantViolation, match="실패 이유"):
        run.fail("  ")

    run.fail("CUDA out of memory (batch_size=512)")
    assert run.status is TrainingStatus.FAILED
    assert "CUDA" in run.failure_reason


def test_전체_과정이_이벤트로_남는다(trained) -> None:
    names = trained.events.names()
    assert "TrainingRunPrepared" in names
    assert "TrainingRunStarted" in names
    assert names.count("EpochCompleted") == 10
    assert "TrainingRunCompleted" in names

    report.block(
        "학습이 남긴 Event",
        "\n".join(
            f"  {name}"
            for name in dict.fromkeys(
                n for n in names if n.startswith(("Training", "Epoch", "Model"))
            )
        ),
    )
