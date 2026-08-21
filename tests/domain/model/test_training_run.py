"""TrainingRun Aggregate — Job 의 생명주기와 승인 판정."""

from __future__ import annotations

import pytest

from domain.model.acceptance import (
    LatencyBudget,
    ModelAcceptancePolicy,
)
from domain.model.architecture import ArchitectureKind, ModelArchitecture
from domain.model.curve import EpochRecord
from domain.model.errors import ModelNotTrained, ShapeMismatch
from domain.model.evaluation import ConfusionMatrix, EvaluationPolicy, EvaluationResult
from domain.model.identifiers import ModelVersionId, TrainingRunId
from domain.model.protocol import EvaluationProtocol, SplitUsage
from domain.model.tensor_spec import DatasetTensorSummary, TensorLayout, TensorSpec
from domain.model.training_config import TrainingConfig
from domain.model.training_data_ref import TrainingDataRef
from domain.model.training_run import TrainingRun, TrainingStatus
from domain.model.windowing import WindowingPlan, WindowLabelPolicy
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.inspection import Verdict

SPEC = TensorSpec(shape=(30, 6), layout=TensorLayout.TIME_FIRST)
ARCHITECTURE = ModelArchitecture(
    kind=ArchitectureKind.CNN1D, input_spec=SPEC, class_count=3
)
PLAN = WindowingPlan(
    window_length=30,
    stride=30,
    label_policy=WindowLabelPolicy(
        priority=(("FAULT", 0.3),), default_label="NORMAL"
    ),
)


def data(**overrides) -> TrainingDataRef:  # noqa: ANN003
    base: dict[str, object] = dict(
        dataset_ref="ds-1",
        uri="power.csv",
        feature_fields=tuple(f"f{i}" for i in range(6)),
        label_field="condition",
        readiness_certified=True,
        quality_gate_passed=True,
    )
    base.update(overrides)
    return TrainingDataRef(**base)  # type: ignore[arg-type]


def new_run(**overrides) -> TrainingRun:  # noqa: ANN003
    return TrainingRun.prepare(
        TrainingRunId.of("run-1"),
        overrides.pop("data", data()),
        overrides.pop("architecture", ARCHITECTURE),
        overrides.pop("config", TrainingConfig(epochs=3)),
        overrides.pop("windowing", PLAN),
        **overrides,
    )


def matrix(fault_correct: int = 8) -> ConfusionMatrix:
    return ConfusionMatrix.from_pairs(
        ("FAULT", "OVERLOAD", "NORMAL"),
        [("FAULT", "FAULT")] * fault_correct
        + [("FAULT", "NORMAL")] * (8 - fault_correct)
        + [("OVERLOAD", "OVERLOAD")] * 42
        + [("NORMAL", "NORMAL")] * 145,
    )


def usage(**overrides) -> SplitUsage:  # noqa: ANN003
    base: dict[str, object] = dict(
        train_sample_count=900,
        validation_sample_count=190,
        test_sample_count=195,
        validation_evaluations=3,
    )
    base.update(overrides)
    return SplitUsage(**base)  # type: ignore[arg-type]


def train_to_completion(run: TrainingRun) -> None:
    run.start()
    run.record_epoch(EpochRecord(1, 1.0, 0.9, 0.4, 0.45))
    run.record_epoch(EpochRecord(2, 0.4, 0.30, 0.85, 0.90))
    run.record_epoch(EpochRecord(3, 0.3, 0.29, 0.88, 0.91))
    run.attach_split_usage(usage())
    run.complete(ModelVersionId.of("mv-1"))


class TestPreparation:
    def test_게이트를_통과하지_않으면_준비조차_안_된다(self) -> None:
        with pytest.raises(IllegalStateTransition, match="게이트를 통과하지 않은"):
            new_run(data=data(quality_gate_passed=False))

    def test_기준을_풀면_준비는_된다(self) -> None:
        run = new_run(data=data(readiness_certified=False), require_gates=False)
        assert run.status is TrainingStatus.PREPARED

    def test_입력_채널_수가_다르면_거부한다(self) -> None:
        with pytest.raises(ShapeMismatch, match="채널 수"):
            new_run(data=data(feature_fields=("a", "b")))

    def test_창_길이와_모델_입력이_다르면_거부한다(self) -> None:
        with pytest.raises(ShapeMismatch, match="창 길이"):
            new_run(
                windowing=WindowingPlan(
                    window_length=60,
                    stride=60,
                    label_policy=PLAN.label_policy,
                )
            )

    def test_텐서_요약의_모양이_다르면_거부한다(self) -> None:
        run = new_run()
        with pytest.raises(ShapeMismatch, match="모양"):
            run.attach_tensor_summary(
                DatasetTensorSummary(split="train", sample_count=10, sample_shape=(60, 6))
            )


class TestLifecycle:
    def test_시작하지_않으면_epoch_을_기록할_수_없다(self) -> None:
        with pytest.raises(IllegalStateTransition, match="시작하지 않은"):
            new_run().record_epoch(EpochRecord(1, 1.0, 1.0, 0.5, 0.5))

    def test_epoch_없이_완료할_수_없다(self) -> None:
        run = new_run()
        run.start()
        with pytest.raises(InvariantViolation, match="epoch 이 하나도 없는"):
            run.complete(ModelVersionId.of("mv-1"))

    def test_완료하면_모델_버전이_생긴다(self) -> None:
        run = new_run()
        train_to_completion(run)
        assert run.status is TrainingStatus.COMPLETED
        assert str(run.model_version_id) == "mv-1"
        assert run.curve.best_epoch.epoch == 3

    def test_끝난_학습을_실패로_바꿀_수_없다(self) -> None:
        run = new_run()
        train_to_completion(run)
        with pytest.raises(IllegalStateTransition, match="이미 끝난"):
            run.fail("뒤늦게 발견")


class TestEvaluationAndAcceptance:
    def test_끝나지_않은_학습은_평가할_수_없다(self) -> None:
        run = new_run()
        with pytest.raises(ModelNotTrained, match="끝나지 않은"):
            run.record_evaluation(EvaluationResult(split="test", matrix=matrix()))

    def test_평가_없이_승인할_수_없다(self) -> None:
        run = new_run()
        train_to_completion(run)
        with pytest.raises(ModelNotTrained, match="평가가 없다"):
            run.accept(ModelAcceptancePolicy())

    def test_분할_기록이_없으면_승인할_수_없다(self) -> None:
        run = new_run()
        run.start()
        run.record_epoch(EpochRecord(1, 1.0, 0.9, 0.4, 0.45))
        run.record_epoch(EpochRecord(2, 0.4, 0.3, 0.85, 0.9))
        run.complete(ModelVersionId.of("mv-1"))
        run.record_evaluation(EvaluationResult(split="test", matrix=matrix()))
        with pytest.raises(ModelNotTrained, match="분할 사용 기록"):
            run.accept(ModelAcceptancePolicy())

    def test_전부_통과하면_ACCEPTED(self) -> None:
        run = new_run()
        train_to_completion(run)
        run.record_evaluation(EvaluationResult(split="test", matrix=matrix()))
        certificate = run.accept(ModelAcceptancePolicy())

        assert certificate.verdict is Verdict.PASSED
        assert run.status is TrainingStatus.ACCEPTED

    def test_분할_경계_누수는_승인을_막는다(self) -> None:
        run = new_run()
        run.start()
        run.record_epoch(EpochRecord(1, 1.0, 0.9, 0.4, 0.45))
        run.record_epoch(EpochRecord(2, 0.4, 0.3, 0.85, 0.9))
        run.attach_split_usage(usage(overlapping_samples=40))
        run.complete(ModelVersionId.of("mv-1"))
        run.record_evaluation(EvaluationResult(split="test", matrix=matrix()))

        certificate = run.accept(ModelAcceptancePolicy())
        assert certificate.is_deployable is False
        assert any(f.code == "PROTOCOL_SPLIT_OVERLAP" for f in certificate.blocking)

    def test_치명_클래스_재현율이_낮으면_막는다(self) -> None:
        run = new_run()
        train_to_completion(run)
        run.record_evaluation(
            EvaluationResult(split="test", matrix=matrix(fault_correct=5))
        )
        certificate = run.accept(
            ModelAcceptancePolicy(
                evaluation=EvaluationPolicy(
                    critical_labels=frozenset({"FAULT"}), min_critical_recall=0.9
                )
            )
        )
        assert certificate.is_deployable is False
        assert any(f.code == "EVAL_RECALL_TOO_LOW" for f in certificate.blocking)

    def test_지연시간_예산을_넘으면_막는다(self) -> None:
        run = new_run()
        train_to_completion(run)
        run.record_evaluation(
            EvaluationResult(split="test", matrix=matrix(), latency_ms_p95=45.0)
        )
        certificate = run.accept(
            ModelAcceptancePolicy(latency=LatencyBudget(p95_ms=30.0))
        )
        assert certificate.is_deployable is False
        assert any(
            f.code == "ACCEPT_LATENCY_OVER_BUDGET" for f in certificate.blocking
        )

    def test_판정_후에는_평가를_덧붙일_수_없다(self) -> None:
        run = new_run()
        train_to_completion(run)
        run.record_evaluation(EvaluationResult(split="test", matrix=matrix()))
        run.accept(ModelAcceptancePolicy())

        with pytest.raises(IllegalStateTransition, match="reopen"):
            run.record_evaluation(EvaluationResult(split="field", matrix=matrix()))

        with pytest.raises(InvariantViolation, match="이유를 남겨야"):
            run.reopen("  ")

        run.reopen("현장 홀드아웃 추가")
        assert run.status is TrainingStatus.COMPLETED
        assert run.certificate is None


class TestAcceptancePolicy:
    def test_게이트_미통과는_승인_단계에서도_다시_막는다(self) -> None:
        run = new_run(data=data(quality_gate_passed=False), require_gates=False)
        train_to_completion(run)
        run.record_evaluation(EvaluationResult(split="test", matrix=matrix()))

        certificate = run.accept(ModelAcceptancePolicy())
        assert any(
            f.code == "ACCEPT_GATES_NOT_PASSED" for f in certificate.blocking
        )

    def test_평가_규율_위반도_승인을_막는다(self) -> None:
        run = new_run()
        run.start()
        run.record_epoch(EpochRecord(1, 1.0, 0.9, 0.4, 0.45))
        run.record_epoch(EpochRecord(2, 0.4, 0.3, 0.85, 0.9))
        run.attach_split_usage(usage(test_evaluations=3))
        run.complete(ModelVersionId.of("mv-1"))
        run.record_evaluation(EvaluationResult(split="test", matrix=matrix()))

        certificate = run.accept(
            ModelAcceptancePolicy(protocol=EvaluationProtocol())
        )
        assert any(f.code == "PROTOCOL_TEST_REUSED" for f in certificate.blocking)
