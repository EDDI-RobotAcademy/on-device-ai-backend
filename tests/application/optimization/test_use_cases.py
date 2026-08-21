"""Optimization Use Case — 조립과 번역.

Use Case 는 판단하지 않는다. 측정기를 부르고, Domain 에 건네고, DTO 로 바꾼다.
그 사실을 확인하기 위해 **측정기를 전부 가짜로 바꿔** 돌린다.
"""

from __future__ import annotations

import pytest

from application.optimization.baseline_mapper import baseline_from
from application.optimization.benchmark_baseline import (
    BenchmarkBaseline,
    BenchmarkBaselineCommand,
)
from application.optimization.compare_candidates import (
    CompareCandidates,
    CompareCandidatesCommand,
)
from application.optimization.convert_model import ConvertModel, ConvertModelCommand
from application.optimization.get_optimization_run import (
    GetOptimizationCertificate,
    GetOptimizationRun,
    GetOptimizationRunQuery,
    GetRooflineProfile,
    ListOptimizationRuns,
)
from application.optimization.profile_roofline import (
    ProfileRoofline,
    ProfileRooflineCommand,
)
from application.optimization.select_model import SelectModel, SelectModelCommand
from application.optimization.start_optimization_run import (
    StartOptimizationRun,
    StartOptimizationRunCommand,
)
from application.shared.errors import ResourceNotFound, UnsupportedOperation
from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.benchmark import BenchmarkResult, MeasurementProtocol
from domain.optimization.conversion import NumericalEquivalence
from domain.optimization.errors import ConversionFailed, OptimizationRunNotFound
from domain.optimization.identifiers import ArtifactId
from domain.optimization.roofline import DeviceCapability, LayerCost, RooflineProfile
from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
from domain.optimization.selection import DeviceBudget
from domain.optimization.tradeoff import ArtifactAccuracy
from infrastructure.persistence.in_memory_optimization_run_repository import (
    InMemoryOptimizationRunRepository,
)

LABELS = ("FAULT", "OVERLOAD", "NORMAL")


# ---------------------------------------------------------------------------
# 가짜 어댑터 — 이 Context 는 자기가 무엇으로 측정되는지 모른다
# ---------------------------------------------------------------------------
class StubExporter:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()

    def supports(self, runtime: RuntimeTarget, precision: Precision) -> bool:
        return not (runtime is RuntimeTarget.TFLITE and precision is Precision.FP16)

    def export(
        self, baseline: BaselineModelRef, runtime: RuntimeTarget, precision: Precision
    ) -> ModelArtifact:
        label = f"{runtime.value}/{precision.value}"
        if label in self.fail_on:
            raise ConversionFailed(f"{label} 를 만들 수 없다", subject=label)
        return ModelArtifact(
            artifact_id=ArtifactId.of(label.lower().replace("/", "-")),
            runtime=runtime,
            precision=precision,
            size_bytes=16_000 // precision.bytes_per_weight,
            uri=f"mem://{label}",
            parameter_count=baseline.parameter_count,
        )


class StubChecker:
    def compare(self, baseline, artifact, sample_count):  # noqa: ANN001, ANN201
        return NumericalEquivalence(
            sample_count=sample_count,
            max_abs_diff=1e-7,
            mean_abs_diff=1e-9,
            agreement_ratio=1.0,
        )


class StubBenchmarker:
    def __init__(self) -> None:
        self.calls: list[MeasurementProtocol] = []

    def benchmark(self, artifact, protocol):  # noqa: ANN001, ANN201
        self.calls.append(protocol)
        base = {"PYTORCH": 2.0, "TORCHSCRIPT": 1.5, "ONNX": 1.0, "TFLITE": 0.5}[
            artifact.runtime.value
        ]
        return BenchmarkResult(
            protocol=protocol,
            p50_ms=base * 0.9,
            p95_ms=base,
            p99_ms=base * 1.1,
            min_ms=base * 0.8,
            max_ms=base * 1.2,
            activation_bytes=8_400,
        )


class StubAccuracy:
    def evaluate(self, baseline, artifact, split):  # noqa: ANN001, ANN201
        drop = 0.03 if artifact.precision is Precision.INT8 else 0.0
        return ArtifactAccuracy(
            accuracy=baseline.accuracy - drop,
            macro_recall=baseline.macro_recall - drop,
            per_class_recall={
                label: value - drop for label, value in baseline.per_class_recall.items()
            },
        )


class StubProfiler:
    def profile(self, baseline, device):  # noqa: ANN001, ANN201
        return RooflineProfile(
            layers=(
                LayerCost(
                    name="conv",
                    kind="Conv1d",
                    mac_count=baseline.mac_count,
                    weight_bytes=baseline.parameter_count * 4,
                    activation_bytes=20_000,
                ),
            ),
            device=device,
        )


class StubTrainingRuns:
    """모듈 3 저장소 자리에 앉는 최소 구현."""

    def __init__(self, run) -> None:  # noqa: ANN001
        self._run = run

    def find_by_id(self, run_id):  # noqa: ANN001, ANN201
        return self._run if str(run_id) == str(self._run.id) else None

    def save(self, run) -> None:  # noqa: ANN001
        self._run = run

    def exists(self, run_id) -> bool:  # noqa: ANN001
        return self.find_by_id(run_id) is not None

    def list_all(self):  # noqa: ANN201
        return (self._run,)


def baseline_ref(**overrides) -> BaselineModelRef:  # noqa: ANN003
    base: dict[str, object] = dict(
        model_version_id="mv-1",
        run_ref="run-1",
        input_shape=(30, 6),
        class_labels=LABELS,
        parameter_count=3_187,
        mac_count=91_296,
        accuracy=0.97,
        macro_recall=0.95,
        per_class_recall={"FAULT": 0.90, "OVERLOAD": 0.96, "NORMAL": 0.99},
        accepted=True,
    )
    base.update(overrides)
    return BaselineModelRef(**base)  # type: ignore[arg-type]


@pytest.fixture
def wiring():  # noqa: ANN201
    from types import SimpleNamespace

    from domain.optimization.identifiers import OptimizationRunId
    from domain.optimization.optimization_run import OptimizationRun

    runs = InMemoryOptimizationRunRepository()
    runs.save(OptimizationRun.start(OptimizationRunId.of("opt-1"), baseline_ref()))

    exporter = StubExporter()
    benchmarker = StubBenchmarker()
    return SimpleNamespace(
        runs=runs,
        exporter=exporter,
        benchmarker=benchmarker,
        checker=StubChecker(),
        accuracy=StubAccuracy(),
        profiler=StubProfiler(),
    )


def benchmark_baseline(wiring, protocol: MeasurementProtocol | None = None):  # noqa: ANN001, ANN201
    return BenchmarkBaseline(
        wiring.runs, wiring.exporter, wiring.benchmarker, wiring.accuracy
    ).execute(
        BenchmarkBaselineCommand(
            run_id="opt-1", protocol=protocol or MeasurementProtocol()
        )
    )


def convert(wiring, runtime: RuntimeTarget, precision=Precision.FP32, **kwargs):  # noqa: ANN001, ANN003, ANN201
    return ConvertModel(
        wiring.runs,
        wiring.exporter,
        wiring.checker,
        wiring.benchmarker,
        wiring.accuracy,
    ).execute(
        ConvertModelCommand(
            run_id="opt-1", runtime=runtime, precision=precision, **kwargs
        )
    )


class Test기준측정:
    def test_프로토콜이_그대로_측정기에_전달된다(self, wiring) -> None:
        protocol = MeasurementProtocol(warmup_runs=5, measured_runs=50)
        view = benchmark_baseline(wiring, protocol)

        assert wiring.benchmarker.calls == [protocol]
        assert "warmup=5" in view.protocol

    def test_기준을_재면_저장된다(self, wiring) -> None:
        benchmark_baseline(wiring)
        from domain.optimization.identifiers import OptimizationRunId

        run = wiring.runs.find_by_id(OptimizationRunId.of("opt-1"))
        assert run.baseline_candidate is not None

    def test_기준_소견은_정책이_만든다(self, wiring) -> None:
        view = benchmark_baseline(
            wiring, MeasurementProtocol(warmup_runs=0, measured_runs=5)
        )
        assert {f.code for f in view.findings} >= {
            "BENCH_NO_WARMUP",
            "BENCH_TOO_FEW_RUNS",
        }


class Test변환:
    def test_변환은_네_가지를_모두_한다(self, wiring) -> None:
        benchmark_baseline(wiring)
        view = convert(wiring, RuntimeTarget.ONNX)

        assert view.size_bytes > 0  # 내보냈다
        assert "argmax 일치" in view.equivalence  # 대조했다
        assert view.p95_ms > 0  # 쟀다
        assert view.accuracy > 0  # 다시 평가했다

    def test_지원하지_않는_조합은_시도하지_않는다(self, wiring) -> None:
        benchmark_baseline(wiring)
        with pytest.raises(UnsupportedOperation):
            convert(wiring, RuntimeTarget.TFLITE, Precision.FP16)

    def test_변환_실패는_기록으로_남고_예외로_올라간다(self, wiring) -> None:
        benchmark_baseline(wiring)
        wiring.exporter.fail_on = {"TFLITE/INT8"}

        with pytest.raises(ConversionFailed):
            convert(wiring, RuntimeTarget.TFLITE, Precision.INT8)

        view = GetOptimizationRun(wiring.runs).execute(
            GetOptimizationRunQuery(run_id="opt-1")
        )
        assert view.rejections and view.rejections[0][0] == "TFLITE/INT8"

    def test_없는_최적화에는_변환할_수_없다(self, wiring) -> None:
        with pytest.raises(OptimizationRunNotFound):
            ConvertModel(
                wiring.runs,
                wiring.exporter,
                wiring.checker,
                wiring.benchmarker,
                wiring.accuracy,
            ).execute(ConvertModelCommand(run_id="없음", runtime=RuntimeTarget.ONNX))


class Test비교와선택:
    def prepared(self, wiring):  # noqa: ANN001, ANN201
        benchmark_baseline(wiring)
        convert(wiring, RuntimeTarget.ONNX)
        convert(wiring, RuntimeTarget.TFLITE, Precision.INT8)
        return wiring

    def test_표는_기준을_포함한다(self, wiring) -> None:
        self.prepared(wiring)
        view = CompareCandidates(wiring.runs).execute(
            CompareCandidatesCommand(run_id="opt-1")
        )
        assert "PYTORCH/FP32" in view.table
        assert "ONNX/FP32" in view.table

    def test_정확도가_떨어진_후보는_예산이_막는다(self, wiring) -> None:
        self.prepared(wiring)
        view = SelectModel(wiring.runs).execute(
            SelectModelCommand(
                run_id="opt-1",
                budget=DeviceBudget(
                    name="설비", latency_p95_ms=3.0, max_accuracy_drop=0.01
                ),
            )
        )
        rejected = {label for label, _ in view.rejected}
        assert "TFLITE/INT8" in rejected
        assert view.selected_label == "ONNX/FP32"

    def test_판정_전에는_판정_기록이_없다(self, wiring) -> None:
        self.prepared(wiring)
        with pytest.raises(ResourceNotFound):
            GetOptimizationCertificate(wiring.runs).execute(
                GetOptimizationRunQuery(run_id="opt-1")
            )


class Test병목:
    def test_프로파일은_붙어_있다가_조회된다(self, wiring) -> None:
        device = DeviceCapability(
            name="edge", peak_gmac_per_second=2.0, memory_bandwidth_gb_per_second=1.6
        )
        ProfileRoofline(wiring.runs, wiring.profiler).execute(
            ProfileRooflineCommand(run_id="opt-1", device=device)
        )
        view = GetRooflineProfile(wiring.runs).execute(
            GetOptimizationRunQuery(run_id="opt-1")
        )
        assert view.total_macs == 91_296
        assert view.machine_balance == pytest.approx(1.25)

    def test_분석하지_않았으면_없다고_말한다(self, wiring) -> None:
        with pytest.raises(ResourceNotFound):
            GetRooflineProfile(wiring.runs).execute(
                GetOptimizationRunQuery(run_id="opt-1")
            )


class Test목록:
    def test_전부_돌려준다(self, wiring) -> None:
        views = ListOptimizationRuns(wiring.runs).execute()
        assert [v.run_id for v in views] == ["opt-1"]


# ---------------------------------------------------------------------------
# ACL — 모듈 3 → 모듈 4 번역
# ---------------------------------------------------------------------------
class Test번역:
    def test_학습이_끝나지_않았으면_최적화할_것이_없다(self) -> None:
        class _Unfinished:
            id = "run-1"
            model_version_id = None

        with pytest.raises(UnsupportedOperation):
            baseline_from(_Unfinished())  # type: ignore[arg-type]

    def test_평가가_없으면_비교할_수_없다(self) -> None:
        class _NoEvaluation:
            id = "run-1"
            model_version_id = "mv-1"

            def evaluation_of(self, split):  # noqa: ANN001, ANN202
                return None

        with pytest.raises(UnsupportedOperation) as caught:
            baseline_from(_NoEvaluation())  # type: ignore[arg-type]
        assert "평가" in str(caught.value)

    def test_시작은_저장소에_남는다(self, wiring) -> None:
        from domain.model.identifiers import TrainingRunId  # noqa: F401

        class _Accepted:
            """모듈 3 의 TrainingRun 자리에 앉는 최소 구현."""

            id = "run-2"
            model_version_id = "mv-2"

            class architecture:  # noqa: N801
                class input_spec:  # noqa: N801
                    shape = (30, 6)

            class profile:  # noqa: N801
                parameter_count = 3_187
                mac_count = 91_296

            @property
            def status(self):  # noqa: ANN201
                from domain.model.training_run import TrainingStatus

                return TrainingStatus.ACCEPTED

            def evaluation_of(self, split):  # noqa: ANN001, ANN202
                from domain.model.evaluation import ConfusionMatrix, EvaluationResult

                matrix = ConfusionMatrix.from_pairs(
                    LABELS,
                    [("FAULT", "FAULT")] * 9
                    + [("FAULT", "NORMAL")]
                    + [("OVERLOAD", "OVERLOAD")] * 20
                    + [("NORMAL", "NORMAL")] * 70,
                )
                return EvaluationResult(split=split, matrix=matrix)

        view = StartOptimizationRun(wiring.runs, StubTrainingRuns(_Accepted())).execute(
            StartOptimizationRunCommand(run_id="opt-2", training_run_id="run-2")
        )
        assert view.status == "OPEN"
        assert view.model_version_id == "mv-2"
        assert len(ListOptimizationRuns(wiring.runs).execute()) == 2
