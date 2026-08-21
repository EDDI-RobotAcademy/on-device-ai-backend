"""OptimizationRun Aggregate 의 불변식.

측정 없이, 파일 없이, 프레임워크 없이 돌아간다.
Domain 이 기술을 모른다는 사실이 여기서 증명된다.
"""

from __future__ import annotations

import pytest

from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.benchmark import BenchmarkResult, MeasurementProtocol
from domain.optimization.conversion import ConversionRecord, NumericalEquivalence
from domain.optimization.errors import NoCandidateSelected
from domain.optimization.identifiers import ArtifactId, OptimizationRunId
from domain.optimization.optimization_run import OptimizationRun, OptimizationStatus
from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
from domain.optimization.selection import DeviceBudget, SelectionPolicy
from domain.optimization.tradeoff import ArtifactAccuracy, OptimizationCandidate
from domain.shared.errors import IllegalStateTransition, InvariantViolation

LABELS = ("FAULT", "OVERLOAD", "NORMAL")


def baseline_ref(*, accepted: bool = True, parameters: int = 3187) -> BaselineModelRef:
    return BaselineModelRef(
        model_version_id="mv-1",
        run_ref="run-1",
        input_shape=(30, 6),
        class_labels=LABELS,
        parameter_count=parameters,
        mac_count=91_296,
        accuracy=0.97,
        macro_recall=0.95,
        per_class_recall={"FAULT": 0.90, "OVERLOAD": 0.96, "NORMAL": 0.99},
        accepted=accepted,
    )


def artifact(
    runtime: RuntimeTarget,
    precision: Precision = Precision.FP32,
    *,
    size: int = 14_000,
    parameters: int = 3187,
) -> ModelArtifact:
    return ModelArtifact(
        artifact_id=ArtifactId.of(f"{runtime.value}-{precision.value}".lower()),
        runtime=runtime,
        precision=precision,
        size_bytes=size,
        uri=f"mem://{runtime.value}",
        parameter_count=parameters,
    )


def benchmark(p95: float = 1.0, *, activation: int = 8_400) -> BenchmarkResult:
    return BenchmarkResult(
        protocol=MeasurementProtocol(),
        p50_ms=p95 * 0.9,
        p95_ms=p95,
        p99_ms=p95 * 1.05,
        min_ms=p95 * 0.8,
        max_ms=p95 * 1.2,
        activation_bytes=activation,
    )


def accuracy(value: float = 0.97, macro: float = 0.95) -> ArtifactAccuracy:
    return ArtifactAccuracy(
        accuracy=value,
        macro_recall=macro,
        per_class_recall={"FAULT": macro, "OVERLOAD": 0.96, "NORMAL": 0.99},
    )


def candidate(
    runtime: RuntimeTarget,
    precision: Precision = Precision.FP32,
    *,
    p95: float = 1.0,
    size: int = 14_000,
    parameters: int = 3187,
    acc: ArtifactAccuracy | None = None,
) -> OptimizationCandidate:
    return OptimizationCandidate(
        artifact=artifact(runtime, precision, size=size, parameters=parameters),
        conversion=ConversionRecord(
            source_runtime=RuntimeTarget.PYTORCH,
            target_runtime=runtime,
            precision=precision,
            equivalence=NumericalEquivalence(
                sample_count=200,
                max_abs_diff=1e-6,
                mean_abs_diff=1e-8,
                agreement_ratio=1.0,
            ),
        ),
        benchmark=benchmark(p95),
        accuracy=acc or accuracy(),
    )


def started(**kwargs) -> OptimizationRun:  # noqa: ANN003
    return OptimizationRun.start(
        OptimizationRunId.of("opt-1"), baseline_ref(**kwargs)
    )


def with_baseline() -> OptimizationRun:
    run = started()
    run.record_baseline(
        artifact(RuntimeTarget.PYTORCH, size=16_273), benchmark(2.0), accuracy()
    )
    return run


class Test시작:
    def test_승인받지_않은_모델은_시작할_수_없다(self) -> None:
        with pytest.raises(IllegalStateTransition):
            OptimizationRun.start(
                OptimizationRunId.of("opt-1"), baseline_ref(accepted=False)
            )

    def test_게이트를_끄면_시작할_수_있다(self) -> None:
        run = OptimizationRun.start(
            OptimizationRunId.of("opt-1"),
            baseline_ref(accepted=False),
            require_accepted=False,
        )
        assert run.status is OptimizationStatus.OPEN

    def test_시작하면_사건이_남는다(self) -> None:
        events = started().pull_events()
        assert [e.event_name for e in events] == ["OptimizationRunStarted"]


class Test기준측정:
    def test_기준을_재기_전에는_후보를_추가할_수_없다(self) -> None:
        run = started()
        with pytest.raises(IllegalStateTransition):
            run.add_candidate(candidate(RuntimeTarget.ONNX))

    def test_기준을_재기_전에는_표를_만들_수_없다(self) -> None:
        with pytest.raises(IllegalStateTransition):
            started().tradeoff_table()

    def test_기준을_재면_상태가_바뀐다(self) -> None:
        run = with_baseline()
        assert run.status is OptimizationStatus.BENCHMARKED
        assert run.baseline_candidate is not None


class Test후보:
    def test_파라미터_수가_다르면_다른_모델이다(self) -> None:
        run = with_baseline()
        with pytest.raises(InvariantViolation) as caught:
            run.add_candidate(candidate(RuntimeTarget.ONNX, parameters=9999))
        assert "다른 모델" in str(caught.value)

    def test_같은_조합은_덮어쓴다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX, p95=1.0))
        run.add_candidate(candidate(RuntimeTarget.ONNX, p95=0.5))

        assert len(run.candidates) == 1
        assert run.candidate_of(RuntimeTarget.ONNX, Precision.FP32).benchmark.p95_ms == 0.5

    def test_실행_경로가_다르면_다른_후보다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX))
        run.add_candidate(candidate(RuntimeTarget.TFLITE))
        run.add_candidate(candidate(RuntimeTarget.TFLITE, Precision.INT8))

        assert len(run.candidates) == 3
        assert run.status is OptimizationStatus.EXPLORING

    def test_이유_없는_실패_기록은_받지_않는다(self) -> None:
        run = with_baseline()
        with pytest.raises(InvariantViolation):
            run.record_rejection("TFLITE/INT8", "   ")

    def test_실패도_기록으로_남는다(self) -> None:
        run = with_baseline()
        run.record_rejection("TFLITE/INT8", "custom_op 를 지원하지 않는다")
        assert run.rejections == (("TFLITE/INT8", "custom_op 를 지원하지 않는다"),)


class Test선택:
    def budget(self, **overrides) -> DeviceBudget:  # noqa: ANN003
        base: dict[str, object] = dict(name="설비", latency_p95_ms=1.5)
        base.update(overrides)
        return DeviceBudget(**base)  # type: ignore[arg-type]

    def test_후보가_없으면_선택할_수_없다(self) -> None:
        run = with_baseline()
        with pytest.raises(NoCandidateSelected):
            run.select(SelectionPolicy(budget=self.budget()))

    def test_예산을_만족하는_것_중_가장_정확한_것을_고른다(self) -> None:
        run = with_baseline()
        run.add_candidate(
            candidate(RuntimeTarget.ONNX, p95=1.0, acc=accuracy(0.97, 0.95))
        )
        run.add_candidate(
            candidate(
                RuntimeTarget.TFLITE, Precision.INT8, p95=0.2, acc=accuracy(0.96, 0.93)
            )
        )

        certificate = run.select(SelectionPolicy(budget=self.budget()))
        # 더 빠른 것은 INT8 이지만, 예산을 이미 만족했으므로 정확한 쪽을 고른다.
        assert certificate.selected_label == "ONNX/FP32"
        assert run.status is OptimizationStatus.SELECTED

    def test_예산을_넘으면_아무것도_고르지_않는다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX, p95=9.0))

        certificate = run.select(SelectionPolicy(budget=self.budget()))
        assert not certificate.has_selection
        assert run.status is OptimizationStatus.BLOCKED

    def test_활성값_예산은_파일_크기와_별개로_막는다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX, size=1_000))

        certificate = run.select(
            SelectionPolicy(budget=self.budget(activation_kib=1.0))
        )
        reasons = " ".join(
            r for v in certificate.verdicts for r in v.reasons
        )
        assert "SELECT_OVER_ACTIVATION_BUDGET" in reasons
        assert not certificate.has_selection

    def test_판정_뒤에는_후보를_추가할_수_없다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX))
        run.select(SelectionPolicy(budget=self.budget()))

        with pytest.raises(IllegalStateTransition):
            run.add_candidate(candidate(RuntimeTarget.TFLITE))

    def test_이유를_남기면_되돌릴_수_있다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX))
        run.select(SelectionPolicy(budget=self.budget()))
        run.reopen("예산이 바뀌었다")

        assert run.status is OptimizationStatus.EXPLORING
        assert run.certificate is None
        run.add_candidate(candidate(RuntimeTarget.TFLITE))

    def test_이유_없이는_되돌릴_수_없다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX))
        run.select(SelectionPolicy(budget=self.budget()))

        with pytest.raises(InvariantViolation):
            run.reopen("")

    def test_판정되지_않은_것은_되돌릴_대상이_아니다(self) -> None:
        with pytest.raises(IllegalStateTransition):
            with_baseline().reopen("아직 판정 전")

    def test_선택하면_사건이_남는다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX))
        run.pull_events()
        run.select(SelectionPolicy(budget=self.budget()))

        assert [e.event_name for e in run.pull_events()] == ["ModelSelected"]

    def test_막히면_막혔다는_사건이_남는다(self) -> None:
        run = with_baseline()
        run.add_candidate(candidate(RuntimeTarget.ONNX, p95=9.0))
        run.pull_events()
        run.select(SelectionPolicy(budget=self.budget()))

        events = run.pull_events()
        assert [e.event_name for e in events] == ["SelectionBlocked"]
        assert events[0].reasons
