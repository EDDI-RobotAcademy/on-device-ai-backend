"""모듈 4 실습 시나리오 빌더.

모듈 4 는 모듈 3 의 **학습된 모델 그 자체**를 필요로 한다.
파일이나 숫자가 아니라 메모리에 올라 있는 모델이다.

그래서 OptimizationContainer 는 모듈 3 컨테이너와
TrainingRunRepository 와 TorchModelRegistry 를 **공유해야 한다.**
따로 만들면 "학습된 모델을 찾을 수 없다"가 된다.
"""

from __future__ import annotations

from pathlib import Path

from application.model.accept_model import AcceptModelCommand
from application.optimization.benchmark_baseline import BenchmarkBaselineCommand
from application.optimization.compare_candidates import (
    CompareCandidatesCommand,
    InspectArtifactSizesCommand,
)
from application.optimization.convert_model import ConvertModelCommand
from application.optimization.profile_roofline import ProfileRooflineCommand
from application.optimization.select_model import SelectModelCommand
from application.optimization.start_optimization_run import (
    StartOptimizationRunCommand,
)
from domain.optimization.benchmark import MeasurementProtocol
from domain.optimization.roofline import DeviceCapability
from domain.optimization.runtime import Precision, RuntimeTarget
from domain.optimization.selection import DeviceBudget, SelectionObjective
from infrastructure.config.container import OptimizationContainer

# 테스트에서는 측정 횟수를 줄인다. 실제 실습에서는 기본값(30/300)을 쓴다.
FAST_PROTOCOL = MeasurementProtocol(warmup_runs=10, measured_runs=100)

CONVERSION_MATRIX: tuple[tuple[RuntimeTarget, Precision], ...] = (
    (RuntimeTarget.TORCHSCRIPT, Precision.FP32),
    (RuntimeTarget.ONNX, Precision.FP32),
    (RuntimeTarget.TFLITE, Precision.FP32),
    (RuntimeTarget.TFLITE, Precision.FP16),
    (RuntimeTarget.TFLITE, Precision.INT8),
)


def edge_device() -> DeviceCapability:
    """전력 감시용 저가 MCU 급 보드 하나. 숫자는 데이터시트에서 온다."""
    return DeviceCapability(
        name="edge-mcu",
        peak_gmac_per_second=2.0,
        memory_bandwidth_gb_per_second=1.6,
    )


def cycle_budget(**overrides) -> DeviceBudget:  # noqa: ANN003
    """설비가 정한 예산. 모델이 정하는 것이 아니다."""
    base: dict[str, object] = dict(
        name="전력 감시 설비",
        latency_p95_ms=1.0,
        storage_kib=64.0,
        activation_kib=64.0,
        min_macro_recall=0.60,
        max_accuracy_drop=0.02,
        max_class_recall_drop=0.10,
    )
    base.update(overrides)
    return DeviceBudget(**base)  # type: ignore[arg-type]


def container_for(trained, artifact_dir: Path) -> OptimizationContainer:  # noqa: ANN001
    return OptimizationContainer.sharing(trained.model, artifact_dir=Path(artifact_dir))


def start(
    optimization: OptimizationContainer,
    *,
    run_id: str,
    training_run_id: str,
    **kwargs,  # noqa: ANN003
):  # noqa: ANN201
    return optimization.start_optimization_run().execute(
        StartOptimizationRunCommand(
            run_id=run_id, training_run_id=training_run_id, **kwargs
        )
    )


def benchmark(
    optimization: OptimizationContainer,
    run_id: str,
    protocol: MeasurementProtocol | None = None,
    **kwargs,  # noqa: ANN003
):  # noqa: ANN201
    return optimization.benchmark_baseline().execute(
        BenchmarkBaselineCommand(
            run_id=run_id, protocol=protocol or FAST_PROTOCOL, **kwargs
        )
    )


def convert(
    optimization: OptimizationContainer,
    run_id: str,
    runtime: RuntimeTarget,
    precision: Precision = Precision.FP32,
    **kwargs,  # noqa: ANN003
):  # noqa: ANN201
    kwargs.setdefault("protocol", FAST_PROTOCOL)
    return optimization.convert_model().execute(
        ConvertModelCommand(
            run_id=run_id, runtime=runtime, precision=precision, **kwargs
        )
    )


def profile(optimization: OptimizationContainer, run_id: str, **kwargs):  # noqa: ANN003, ANN201
    kwargs.setdefault("device", edge_device())
    return optimization.profile_roofline().execute(
        ProfileRooflineCommand(run_id=run_id, **kwargs)
    )


def compare(optimization: OptimizationContainer, run_id: str):  # noqa: ANN201
    return optimization.compare_candidates().execute(
        CompareCandidatesCommand(run_id=run_id)
    )


def sizes(optimization: OptimizationContainer, run_id: str):  # noqa: ANN201
    return optimization.inspect_artifact_sizes().execute(
        InspectArtifactSizesCommand(run_id=run_id)
    )


def select(
    optimization: OptimizationContainer,
    run_id: str,
    budget: DeviceBudget | None = None,
    objective: SelectionObjective = SelectionObjective.ACCURACY,
    **kwargs,  # noqa: ANN003
):  # noqa: ANN201
    return optimization.select_model().execute(
        SelectModelCommand(
            run_id=run_id,
            budget=budget or cycle_budget(),
            objective=objective,
            **kwargs,
        )
    )


def build_accepted_model(train_path: Path):  # noqa: ANN201
    """모듈 4 전용 학습 파이프라인.

    창을 **겹치지 않게**(stride = window_length) 자른다.
    겹치게 자르면 분할 사이에 같은 표본이 새고, 모듈 3 의 승인 판정에서 막힌다 (실습 3-8).
    승인받지 않은 모델은 최적화 대상이 아니므로 (실습 4-1) 여기서부터 지켜야 한다.
    """
    from tests.support.model_scenario import build_pipeline as build_model

    return build_model(
        dataset_id="power-opt",
        assessment_id="qa-power-opt",
        run_id="run-opt",
        train_path=train_path,
        stride=30,
    )


def build_pipeline(trained, artifact_dir: Path, *, run_id: str = "opt-power"):  # noqa: ANN001, ANN201
    """모듈 3 승인 → 최적화 시작 → 기준 측정 → 전 조합 변환 → 병목 분석.

    선택(4-10)은 **일부러 여기서 하지 않는다.**
    예산에 따라 답이 달라지는 것을 실습에서 직접 보여야 하기 때문이다.
    """
    from types import SimpleNamespace

    from domain.model.training_run import TrainingStatus

    training_run_id = trained.run_id
    training_run = trained.model.runs.find_by_id(_training_id(training_run_id))
    if training_run.status is not TrainingStatus.ACCEPTED:
        trained.model.accept_model().execute(AcceptModelCommand(run_id=training_run_id))

    optimization = container_for(trained, artifact_dir)
    start(optimization, run_id=run_id, training_run_id=training_run_id)
    baseline = benchmark(optimization, run_id)

    candidates = {}
    rejections = {}
    for runtime, precision in CONVERSION_MATRIX:
        try:
            candidates[f"{runtime.value}/{precision.value}"] = convert(
                optimization, run_id, runtime, precision
            )
        except Exception as exc:  # noqa: BLE001 - 실패도 결과다 (실습 4-4)
            rejections[f"{runtime.value}/{precision.value}"] = str(exc)

    roofline = profile(optimization, run_id)

    return SimpleNamespace(
        optimization=optimization,
        trained=trained,
        run_id=run_id,
        training_run_id=training_run_id,
        baseline=baseline,
        candidates=candidates,
        rejections=rejections,
        roofline=roofline,
        table=compare(optimization, run_id),
    )


def _training_id(run_id: str):  # noqa: ANN202
    from domain.model.identifiers import TrainingRunId

    return TrainingRunId.of(run_id)


# ---------------------------------------------------------------------------
# 구조 축소 (실습 4-11)
# ---------------------------------------------------------------------------
REDUCTIONS = (
    ("폭 절반 · 재학습 없음", "WIDTH", 0.5, False),
    ("폭 절반 · 재학습", "WIDTH", 0.5, True),
    ("가지치기 50% · 그대로", "PRUNE_UNSTRUCTURED", 0.5, False),
    ("가지치기 50% · 미세조정", "PRUNE_UNSTRUCTURED", 0.5, True),
    ("채널 가지치기 50%", "PRUNE_STRUCTURED", 0.5, False),
)


def reduce_structure(optimization_container, run_id: str):  # noqa: ANN001, ANN201
    """다섯 가지 축소를 한 번에 적용해 비교표를 만든다."""
    from application.optimization.reduce_structure import ReduceStructureCommand
    from domain.optimization.structural import ReductionKind, StructuralReduction

    return optimization_container.reduce_structure().execute(
        ReduceStructureCommand(
            run_id=run_id,
            fine_tune_epochs=25,
            reductions=tuple(
                (
                    label,
                    StructuralReduction(
                        kind=ReductionKind[kind], ratio=ratio, fine_tuned=tuned
                    ),
                )
                for label, kind, ratio, tuned in REDUCTIONS
            ),
        )
    )
