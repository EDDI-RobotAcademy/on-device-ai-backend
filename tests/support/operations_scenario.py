"""모듈 5 실습 시나리오 빌더.

모듈 5 는 앞의 네 모듈 **전부** 위에 선다.

    모듈 1·2 게이트 통과 → 모듈 3 학습·승인 → 모듈 4 변환·선택 → 모듈 5 배포·관측

그리고 여기서 처음으로 **시간이 흐른다.**
4일치 현장 신호를 몇 초에 재생하고, 창 단위로 지켜본다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from application.operations.compare_shadow import CompareShadowCommand
from application.operations.decide_retraining import DecideRetrainingCommand
from application.operations.deploy_model import (
    DeployModelCommand,
    ReleaseVersionCommand,
)
from application.operations.find_onset import FindOnsetQuery
from application.operations.observe_health import (
    IngestInferenceLogCommand,
    ObserveHealthCommand,
)
from application.operations.respond_to_incident import (
    QuarantineCommand,
    RollbackCommand,
)
from domain.operations.identifiers import DeploymentId
from domain.operations.latency import LatencyPolicy
from domain.operations.prediction_mix import PredictionDriftPolicy
from domain.operations.target import DeploymentTarget, TargetKind
from infrastructure.config.container import OperationsContainer
from infrastructure.monitoring.inference_log_store import slice_windows
from infrastructure.sample_data.plant_power_operations import WINDOW_HOURS

DEPLOYMENT_ID = "dep-line3"
WATCH_ID = "watch-dep-line3"
DEPLOY_MOMENT = "2026-05-19 23:00:00"
"""현장 신호가 시작되기 전. 배포가 먼저고 관측이 그 다음이다."""

CYCLE_BUDGET_MS = 1.0
"""설비 사이클 타임. 모듈 4 의 DeviceBudget 과 같은 숫자다."""


def target(kind: TargetKind = TargetKind.DEVICE_GROUP, count: int = 3) -> DeploymentTarget:
    return DeploymentTarget(
        kind=kind, identifier="LINE-3", name="3라인 전력 감시", device_count=count
    )


def latency_policy(**overrides) -> LatencyPolicy:  # noqa: ANN003
    base: dict[str, object] = dict(
        cycle_budget_ms=CYCLE_BUDGET_MS,
        max_regression_ratio=12.0,
        max_jitter_ratio=3.0,
    )
    base.update(overrides)
    return LatencyPolicy(**base)  # type: ignore[arg-type]


def mix_policy(**overrides) -> PredictionDriftPolicy:  # noqa: ANN003
    base: dict[str, object] = dict(
        max_shift=0.15, critical_labels=frozenset({"FAULT"})
    )
    base.update(overrides)
    return PredictionDriftPolicy(**base)  # type: ignore[arg-type]


def deploy(operations: OperationsContainer, optimized, trained, **overrides):  # noqa: ANN001, ANN003, ANN201
    body: dict[str, object] = dict(
        deployment_id=DEPLOYMENT_ID,
        optimization_run_id=optimized.run_id,
        training_run_id=trained.run_id,
        target=target(),
        watch_id=WATCH_ID,
        released_at=DEPLOY_MOMENT,
        note="첫 배포",
    )
    body.update(overrides)
    return operations.deploy_model().execute(DeployModelCommand(**body))  # type: ignore[arg-type]


def ingest(operations: OperationsContainer, records, deployment_id: str = DEPLOYMENT_ID):  # noqa: ANN001, ANN201
    operations.logs.bind(DeploymentId.of(deployment_id))
    return operations.ingest_inference_log().execute(
        IngestInferenceLogCommand(
            deployment_id=deployment_id, records=tuple(records)
        )
    )


def windows(operations: OperationsContainer, deployment_id: str = DEPLOYMENT_ID, **kwargs):  # noqa: ANN003, ANN201
    records = operations.logs.all_records(DeploymentId.of(deployment_id))
    return slice_windows(records, hours=WINDOW_HOURS, **kwargs)


def observe(operations: OperationsContainer, window, **overrides):  # noqa: ANN001, ANN003, ANN201
    body: dict[str, object] = dict(
        deployment_id=DEPLOYMENT_ID,
        window=window,
        latency_policy=latency_policy(),
        mix_policy=mix_policy(),
    )
    body.update(overrides)
    return operations.observe_health().execute(ObserveHealthCommand(**body))  # type: ignore[arg-type]


def observe_all(operations: OperationsContainer, **overrides):  # noqa: ANN003, ANN201
    return [observe(operations, w, **overrides) for w in windows(operations)]


def rebaseline(operations: OperationsContainer, window, reason: str = "배포 직후 안정 구간"):  # noqa: ANN001, ANN201
    from application.operations.observe_health import RebaselineCommand

    return operations.rebaseline_watch().execute(
        RebaselineCommand(
            deployment_id=DEPLOYMENT_ID, window=window, reason=reason
        )
    )


def onset(operations: OperationsContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(watch_id=WATCH_ID)
    body.update(overrides)
    return operations.find_onset().execute(FindOnsetQuery(**body))  # type: ignore[arg-type]


def quarantine(operations: OperationsContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(deployment_id=DEPLOYMENT_ID)
    body.update(overrides)
    return operations.quarantine_deployment().execute(QuarantineCommand(**body))  # type: ignore[arg-type]


def rollback(operations: OperationsContainer, to_version: int, reason: str, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        deployment_id=DEPLOYMENT_ID, to_version=to_version, reason=reason
    )
    body.update(overrides)
    return operations.rollback_deployment().execute(RollbackCommand(**body))  # type: ignore[arg-type]


def release(operations: OperationsContainer, optimized, trained, **overrides):  # noqa: ANN001, ANN003, ANN201
    body: dict[str, object] = dict(
        deployment_id=DEPLOYMENT_ID,
        optimization_run_id=optimized.run_id,
        training_run_id=trained.run_id,
    )
    body.update(overrides)
    return operations.release_version().execute(ReleaseVersionCommand(**body))  # type: ignore[arg-type]


def compare_shadow(operations: OperationsContainer, window, artifact_id: str, **overrides):  # noqa: ANN001, ANN003, ANN201
    body: dict[str, object] = dict(
        deployment_id=DEPLOYMENT_ID,
        window=window,
        candidate_artifact_id=artifact_id,
    )
    body.update(overrides)
    return operations.compare_shadow().execute(CompareShadowCommand(**body))  # type: ignore[arg-type]


def decide_retraining(operations: OperationsContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(watch_id=WATCH_ID)
    body.update(overrides)
    return operations.decide_retraining().execute(DecideRetrainingCommand(**body))  # type: ignore[arg-type]


def attach_shadow(operations, deployed, candidate_label: str = "TFLITE/INT8"):  # noqa: ANN001, ANN201
    """그림자 실행기를 조립한다. (실습 5-9)

    새 결과물을 **같은 4일치 입력에 실제로 다시 돌린다.**
    현장에서는 디바이스가 두 모델을 동시에 돌리지만, 여기서는 같은 입력을 재생한다 —
    결과는 같다.
    """
    from domain.model.identifiers import ModelVersionId, TrainingRunId
    from domain.operations.identifiers import DeploymentId
    from domain.optimization.identifiers import ArtifactId, OptimizationRunId
    from infrastructure.edge.device_simulator import (
        DeviceFleetSimulator,
        SimulationSpec,
    )
    from infrastructure.monitoring.field_measurers import ReplayShadowRunner

    optimization = deployed.optimized.optimization
    run = optimization.runs.find_by_id(OptimizationRunId.of(deployed.optimized.run_id))
    candidate = next(
        c for c in run.tradeoff_table().all_candidates if c.label == candidate_label
    )
    loaded = optimization.runtimes.require(
        ArtifactId.of(str(candidate.artifact.artifact_id))
    )

    training_run = operations.training_runs.find_by_id(
        TrainingRunId.of(deployed.trained.run_id)
    )
    model = optimization.registry.get(
        ModelVersionId.of(run.baseline.model_version_id)
    )
    simulator = DeviceFleetSimulator(
        SimulationSpec(
            stream_uri=str(deployed.stream),
            feature_fields=tuple(training_run.data.feature_fields),
            label_field="condition",
            window_length=training_run.windowing.window_length,
            stride=3,
            class_labels=model.dataset.labels,
            normalization=dict(training_run.data.normalization),
            baseline_p95_ms=candidate.benchmark.p95_ms,
        )
    )
    shadow_records = simulator.run(loaded.predict, deployment_version=1)
    by_moment = {(r.occurred_at, r.device_id): r for r in shadow_records}

    def replay(window, artifact_id):  # noqa: ANN001, ANN202
        incumbent = operations.logs.records_in(
            DeploymentId.of(DEPLOYMENT_ID), window
        )
        return [
            by_moment[(r.occurred_at, r.device_id)].predicted_label
            for r in incumbent
            if (r.occurred_at, r.device_id) in by_moment
        ]

    operations.shadow = ReplayShadowRunner(
        operations.logs,
        replay=replay,
        candidate_latency_ms=candidate.benchmark.p95_ms * 8.0,
        incumbent_label=deployed.deploy_result.deployment.current_artifact,
        candidate_label=str(candidate.artifact.artifact_id),
    )
    return str(candidate.artifact.artifact_id)


def clone(deployed, *, deployment_id: str = DEPLOYMENT_ID):  # noqa: ANN001, ANN201
    """상태를 바꿔도 되는 사본.

    로그와 측정기는 세션 것을 **공유한다** — 34,533건을 다시 만들지 않기 위해서다.
    저장소만 새로 만들고 같은 ID 로 다시 배포한다.
    """
    from infrastructure.persistence.in_memory_deployment_repository import (
        InMemoryDeploymentRepository,
        InMemoryHealthWatchRepository,
    )

    source = deployed.operations
    operations = OperationsContainer(
        optimization_runs=source.optimization_runs,
        training_runs=source.training_runs,
        deployments=InMemoryDeploymentRepository(),
        watches=InMemoryHealthWatchRepository(),
        logs=source.logs,
        publisher=source.publisher,
        drift=source.drift,
    )
    deploy(
        operations,
        deployed.optimized,
        deployed.trained,
        deployment_id=deployment_id,
    )
    return operations


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------
def build_pipeline(
    optimized, trained, stream: Path, *, observe_windows: bool = True, rebase: bool = True
):  # noqa: ANN001, ANN201
    """배포 → 4일치 재생 → 창마다 관측.

    판정(격리·롤백·재학습)은 **여기서 하지 않는다.**
    실습마다 다른 기준으로 판단해 보아야 하기 때문이다.
    """
    from infrastructure.config.container import OperationsContainer
    from infrastructure.monitoring.field_measurers import StreamInputDriftMeasurer

    from domain.model.identifiers import ModelVersionId, TrainingRunId
    from domain.optimization.optimization_run import OptimizationStatus
    from domain.optimization.identifiers import OptimizationRunId

    # 모듈 5 는 모듈 4 의 **선택**을 전제로 한다. 아직 안 골랐으면 여기서 고른다.
    optimization_run = optimized.optimization.runs.find_by_id(
        OptimizationRunId.of(optimized.run_id)
    )
    if optimization_run.status is not OptimizationStatus.SELECTED:
        from tests.support import optimization_scenario as os4

        os4.select(optimized.optimization, optimized.run_id)

    operations = OperationsContainer.sharing(optimized.optimization)
    result = deploy(operations, optimized, trained)

    # 학습이 무엇을 어떻게 먹었는지는 TrainingRun 이 들고 있다.
    training_run = operations.training_runs.find_by_id(
        TrainingRunId.of(trained.run_id)
    )
    feature_fields = tuple(training_run.data.feature_fields)

    # 입력 드리프트 측정기는 **학습 분포**를 알아야 만들 수 있다 (실습 5-7).
    operations.drift = StreamInputDriftMeasurer(
        str(stream),
        feature_fields=feature_fields,
        reference=_reference_columns(training_run.data.uri, feature_fields),
    )

    model = optimized.optimization.registry.get(
        ModelVersionId.of(result.deployment.model_version_id)
    )
    records = _replay(operations, optimized, training_run, stream, result, model)
    ingest(operations, records)

    # 기준 재고정 (실습 5-6).
    # 평가 데이터의 예측 분포와 현장의 예측 분포는 처음부터 다르다.
    # 배포 직후 아무 일도 없었던 첫 창을 새 기준으로 못박는다.
    all_windows = windows(operations)
    if rebase and all_windows:
        rebaseline(operations, all_windows[0])

    reports = observe_all(operations) if observe_windows else []
    return SimpleNamespace(
        operations=operations,
        optimized=optimized,
        trained=trained,
        deployment_id=DEPLOYMENT_ID,
        watch_id=WATCH_ID,
        deploy_result=result,
        records=records,
        reports=reports,
        stream=stream,
        feature_fields=feature_fields,
    )


def _replay(operations, optimized, training_run, stream, result, model):  # noqa: ANN001, ANN202
    """모듈 4 가 고른 결과물을 4일치 현장 신호에 **실제로** 돌린다.

    정규화 통계는 **학습 때 쓰던 것을 그대로** 쓴다 (실습 1-7, 5-1).
    여기서 다시 계산하면 디바이스가 다른 전처리를 하는 것이 된다.
    """
    from domain.optimization.identifiers import ArtifactId
    from infrastructure.edge.device_simulator import (
        DeviceFleetSimulator,
        SimulationSpec,
    )

    artifact_id = result.deployment.current_artifact
    loaded = optimized.optimization.runtimes.require(ArtifactId.of(artifact_id))

    spec = SimulationSpec(
        stream_uri=str(stream),
        feature_fields=tuple(training_run.data.feature_fields),
        label_field="condition",
        window_length=training_run.windowing.window_length,
        # 디바이스는 매 표본마다 판단하지 않는다. 30초에 한 번이면 충분하다.
        # 학습(실습 3-8)과 달리 겹쳐도 된다 — 분할을 하지 않으니 샐 것이 없다.
        stride=3,
        class_labels=model.dataset.labels,
        normalization=dict(training_run.data.normalization),
        baseline_p95_ms=0.003,
    )
    return DeviceFleetSimulator(spec).run(loaded.predict, deployment_version=1)


def _reference_columns(uri: str, feature_fields):  # noqa: ANN001, ANN202
    """학습 데이터의 **원본 값** 분포. 정규화 전 값이어야 현장과 견줄 수 있다."""
    from infrastructure.analysis.table_loader import load_frame, numeric_view

    frame = load_frame(uri, "CSV").frame
    return {
        name: numeric_view(frame[name]).to_numpy(dtype="float64")
        for name in feature_fields
        if name in frame.columns
    }


# ---------------------------------------------------------------------------
# 디바이스 파이프라인 (실습 5-12, 5-13)
# ---------------------------------------------------------------------------
def build_device_pipeline(optimized, trained, stream, *, device_id: str = "DEV-01"):  # noqa: ANN001, ANN201
    """디바이스 안에서 도는 다섯 단계를 실제로 돌린다."""
    from types import SimpleNamespace

    from domain.model.identifiers import TrainingRunId
    from domain.operations.alerting import AlertRule
    from domain.operations.pipeline import PipelineContract
    from domain.optimization.identifiers import OptimizationRunId
    from domain.optimization.optimization_run import OptimizationStatus
    from infrastructure.edge.pipeline_runner import DevicePipelineRunner, PipelineSpec

    optimization_run = optimized.optimization.runs.find_by_id(
        OptimizationRunId.of(optimized.run_id)
    )
    if optimization_run.status is not OptimizationStatus.SELECTED:
        from tests.support import optimization_scenario as os4

        os4.select(optimized.optimization, optimized.run_id)
        optimization_run = optimized.optimization.runs.find_by_id(
            OptimizationRunId.of(optimized.run_id)
        )

    training_run = optimized.optimization.training_runs.find_by_id(
        TrainingRunId.of(trained.run_id)
    )
    model = optimized.optimization.registry.get(training_run.model_version_id)
    selected = optimization_run.certificate.selected_artifact_id
    loaded = optimized.optimization.runtimes.require(selected)

    contract = PipelineContract(
        input_shape=tuple(training_run.architecture.input_spec.shape),
        sample_interval_seconds=10.0,
        feature_fields=tuple(training_run.data.feature_fields),
        normalization=dict(training_run.data.normalization),
        class_labels=model.dataset.labels,
    )
    rule = AlertRule(
        alert_labels=("FAULT", "OVERLOAD"),
        dwell=3,
        min_confidence=0.6,
        cooldown_seconds=300.0,
        hourly_budget=12,
    )
    runner = DevicePipelineRunner(
        PipelineSpec(
            stream_uri=str(stream), device_id=device_id, contract=contract
        )
    )
    run, ledger = runner.run(loaded.predict, rule)
    return SimpleNamespace(
        run=run, ledger=ledger, contract=contract, rule=rule,
        predict=loaded.predict, spec_stream=str(stream), device_id=device_id,
    )
