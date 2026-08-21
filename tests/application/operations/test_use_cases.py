"""Operations Use Case — 조립과 번역.

측정기를 전부 가짜로 바꿔 돌린다.
**Use Case 가 판단하지 않는다**는 사실이 이렇게 증명된다.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from application.operations.decide_retraining import (
    DecideRetraining,
    DecideRetrainingCommand,
)
from application.operations.deploy_model import (
    DeployModel,
    DeployModelCommand,
    GetDeployment,
    GetDeploymentQuery,
    ListDeployments,
)
from application.operations.find_onset import (
    FindOnset,
    FindOnsetQuery,
    GetTimeline,
    GetTimelineQuery,
    GetWatchForDeployment,
    GetWatchForDeploymentQuery,
)
from application.operations.observe_health import (
    IngestInferenceLog,
    IngestInferenceLogCommand,
    ObserveHealth,
    ObserveHealthCommand,
    RebaselineCommand,
    RebaselineWatch,
)
from application.operations.respond_to_incident import (
    QuarantineCommand,
    QuarantineDeployment,
    RollbackCommand,
    RollbackDeployment,
)
from application.shared.errors import ConflictingRequest
from domain.operations.drift import DriftReport, FeatureDrift
from domain.operations.errors import DeploymentNotFound, HealthWatchNotFound
from domain.operations.health import HealthMetric
from domain.operations.identifiers import DeploymentId
from domain.operations.inference_log import InferenceRecord
from domain.operations.latency import LatencyPolicy, LatencyProfile
from domain.operations.prediction_mix import PredictionMix
from domain.operations.target import DeploymentTarget, TargetKind
from domain.operations.window import ObservationWindow
from infrastructure.monitoring.inference_log_store import InMemoryInferenceLogStore
from infrastructure.persistence.in_memory_deployment_repository import (
    InMemoryDeploymentRepository,
    InMemoryHealthWatchRepository,
)

LABELS = ("FAULT", "OVERLOAD", "NORMAL")
BASELINE_MIX = {"NORMAL": 0.78, "OVERLOAD": 0.21, "FAULT": 0.01}


class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 5, 19, 23, 0, 0, tzinfo=UTC)


class StubOptimizationRuns:
    """모듈 4 저장소 자리에 앉는 최소 구현."""

    def __init__(self, run) -> None:  # noqa: ANN001
        self._run = run

    def find_by_id(self, run_id):  # noqa: ANN001, ANN201
        return self._run if str(run_id) == "opt-1" else None

    def save(self, run) -> None:  # noqa: ANN001
        self._run = run

    def exists(self, run_id) -> bool:  # noqa: ANN001
        return self.find_by_id(run_id) is not None

    def list_all(self):  # noqa: ANN201
        return (self._run,)


class StubTrainingRuns:
    def __init__(self, run) -> None:  # noqa: ANN001
        self._run = run

    def find_by_id(self, run_id):  # noqa: ANN001, ANN201
        return self._run if str(run_id) == "run-1" else None

    def save(self, run) -> None:  # noqa: ANN001
        self._run = run

    def exists(self, run_id) -> bool:  # noqa: ANN001
        return self.find_by_id(run_id) is not None

    def list_all(self):  # noqa: ANN201
        return (self._run,)


class StubLatency:
    def __init__(self, p95: float = 0.03) -> None:
        self.p95 = p95

    def measure(self, deployment_id, window):  # noqa: ANN001, ANN201
        return LatencyProfile(
            window=window,
            p50_ms=self.p95 * 0.6,
            p95_ms=self.p95,
            p99_ms=self.p95 * 1.4,
            max_ms=self.p95 * 3,
        )


class StubMix:
    def __init__(self, counts=None) -> None:  # noqa: ANN001
        self.counts = counts or {"NORMAL": 780, "OVERLOAD": 210, "FAULT": 10}

    def measure(self, deployment_id, window):  # noqa: ANN001, ANN201
        return PredictionMix(
            window=window,
            counts=self.counts,
            mean_confidence={label: 0.95 for label in self.counts},
        )


class StubDrift:
    def __init__(self, psi: float = 0.01) -> None:
        self.psi = psi

    def measure(self, deployment_id, window):  # noqa: ANN001, ANN201
        return DriftReport(
            window=window,
            features=(
                FeatureDrift(
                    field_name="temperature_c",
                    psi=self.psi,
                    mean_shift_sigma=self.psi,
                    out_of_range_ratio=0.0,
                ),
            ),
        )


def optimization_run():  # noqa: ANN201
    """모듈 4 의 OptimizationRun 자리에 앉는 최소 구현."""
    from types import SimpleNamespace

    from domain.optimization.benchmark import BenchmarkResult, MeasurementProtocol
    from domain.optimization.conversion import ConversionRecord
    from domain.optimization.identifiers import ArtifactId
    from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
    from domain.optimization.tradeoff import (
        ArtifactAccuracy,
        OptimizationCandidate,
        TradeoffTable,
    )

    candidate = OptimizationCandidate(
        artifact=ModelArtifact(
            artifact_id=ArtifactId.of("mv-1-tflite-fp16"),
            runtime=RuntimeTarget.TFLITE,
            precision=Precision.FP16,
            size_bytes=11_724,
            uri="mem://a",
            parameter_count=3_187,
        ),
        conversion=ConversionRecord(
            source_runtime=RuntimeTarget.PYTORCH,
            target_runtime=RuntimeTarget.TFLITE,
            precision=Precision.FP16,
        ),
        benchmark=BenchmarkResult(
            protocol=MeasurementProtocol(),
            p50_ms=0.0030,
            p95_ms=0.0031,
            p99_ms=0.0033,
            min_ms=0.0028,
            max_ms=0.0040,
        ),
        accuracy=ArtifactAccuracy(
            accuracy=0.97,
            macro_recall=0.95,
            per_class_recall={"FAULT": 0.9, "OVERLOAD": 0.96, "NORMAL": 0.99},
            predicted_mix=BASELINE_MIX,
        ),
    )
    baseline = SimpleNamespace(model_version_id="mv-1", class_labels=LABELS)
    certificate = SimpleNamespace(
        has_selection=True, selected_label="TFLITE/FP16"
    )
    return SimpleNamespace(
        id="opt-1",
        baseline=baseline,
        certificate=certificate,
        tradeoff_table=lambda: TradeoffTable(
            baseline=candidate, candidates=(candidate,)
        ),
    )


def training_run():  # noqa: ANN201
    from types import SimpleNamespace

    return SimpleNamespace(
        id="run-1",
        data=SimpleNamespace(
            feature_fields=("active_power_kw", "temperature_c"),
            normalization={"active_power_kw": (147.8, 39.8)},
            uri="mem://train.csv",
        ),
    )


def window(index: int = 0, samples: int = 2880) -> ObservationWindow:
    day, hour = 20 + index // 3, (index % 3) * 8
    return ObservationWindow(
        label=f"05-{day:02d} {hour:02d}시",
        started_at=f"2026-05-{day:02d} {hour:02d}:00:00",
        ended_at=f"2026-05-{day:02d} {hour + 7:02d}:59:59",
        sample_count=samples,
    )


def records(index: int, count: int = 300, *, label: str = "NORMAL"):  # noqa: ANN201
    day, hour = 20 + index // 3, (index % 3) * 8
    return tuple(
        InferenceRecord(
            occurred_at=f"2026-05-{day:02d} {hour:02d}:{n // 60:02d}:{n % 60:02d}",
            device_id=f"DEV-{n % 3 + 1:02d}",
            deployment_version=1,
            predicted_label=label,
            confidence=0.95,
            latency_ms=0.03,
            input_digest=f"d{index}-{n}",
            ground_truth=label if n % 8 == 0 else None,
        )
        for n in range(count)
    )


@pytest.fixture
def wiring():  # noqa: ANN201
    from types import SimpleNamespace

    return SimpleNamespace(
        deployments=InMemoryDeploymentRepository(),
        watches=InMemoryHealthWatchRepository(),
        logs=InMemoryInferenceLogStore(),
        optimization_runs=StubOptimizationRuns(optimization_run()),
        training_runs=StubTrainingRuns(training_run()),
        clock=FixedClock(),
        latency=StubLatency(),
        mix=StubMix(),
        drift=StubDrift(),
    )


def deploy(wiring, **overrides):  # noqa: ANN001, ANN003, ANN201
    body: dict[str, object] = dict(
        deployment_id="dep-1",
        optimization_run_id="opt-1",
        training_run_id="run-1",
        target=DeploymentTarget(
            kind=TargetKind.DEVICE_GROUP, identifier="LINE-3", device_count=3
        ),
        released_at="2026-05-19 23:00:00",
    )
    body.update(overrides)
    return DeployModel(
        wiring.deployments,
        wiring.watches,
        wiring.optimization_runs,
        wiring.training_runs,
        wiring.clock,
    ).execute(DeployModelCommand(**body))  # type: ignore[arg-type]


def observe(wiring, index: int, **overrides):  # noqa: ANN001, ANN003, ANN201
    body: dict[str, object] = dict(
        deployment_id="dep-1",
        window=window(index),
        latency_policy=LatencyPolicy(cycle_budget_ms=30.0, max_regression_ratio=20.0),
    )
    body.update(overrides)
    return ObserveHealth(
        wiring.deployments,
        wiring.watches,
        wiring.logs,
        wiring.latency,
        wiring.mix,
        wiring.drift,
    ).execute(ObserveHealthCommand(**body))  # type: ignore[arg-type]


def ingest(wiring, index: int, **kwargs):  # noqa: ANN001, ANN003, ANN201
    wiring.logs.bind(DeploymentId.of("dep-1"))
    return IngestInferenceLog(wiring.deployments, wiring.logs).execute(
        IngestInferenceLogCommand(
            deployment_id="dep-1", records=records(index, **kwargs)
        )
    )


class Test배포:
    def test_배포와_동시에_관측이_열린다(self, wiring) -> None:
        result = deploy(wiring)
        assert result.watch_id == "watch-dep-1"
        assert wiring.watches.find_by_id(_watch_id(result.watch_id)) is not None

    def test_전처리_통계가_함께_넘어온다(self, wiring) -> None:
        """**모듈 3 에서 가져온다.** 결과물 파일에는 들어 있지 않다."""
        deploy(wiring)
        deployment = wiring.deployments.find_by_id(DeploymentId.of("dep-1"))
        assert deployment.current_version.artifact.normalization

    def test_학습을_안_넘기면_전처리를_알_수_없다(self, wiring) -> None:
        """정규화 통계도, 입력 채널 목록도 모듈 3 에서 온다."""
        result = deploy(wiring, training_run_id=None)
        assert not result.check.can_release
        assert "RELEASE_NO_INPUT_SCHEMA" in {f.code for f in result.check.findings}

    def test_기준_숫자는_모듈4_에서_온다(self, wiring) -> None:
        result = deploy(wiring)
        watch = wiring.watches.find_by_id(_watch_id(result.watch_id))
        assert watch.baseline_p95_ms == pytest.approx(0.0031)
        assert watch.baseline_mix == BASELINE_MIX

    def test_없는_최적화는_없다고_말한다(self, wiring) -> None:
        from domain.optimization.errors import OptimizationRunNotFound

        with pytest.raises(OptimizationRunNotFound):
            deploy(wiring, optimization_run_id="없음")

    def test_목록과_조회(self, wiring) -> None:
        deploy(wiring)
        assert [v.deployment_id for v in ListDeployments(wiring.deployments).execute()] == [
            "dep-1"
        ]
        view = GetDeployment(wiring.deployments).execute(
            GetDeploymentQuery(deployment_id="dep-1")
        )
        assert view.current_version == 1

    def test_없는_배포_조회는_막힌다(self, wiring) -> None:
        with pytest.raises(DeploymentNotFound):
            GetDeployment(wiring.deployments).execute(
                GetDeploymentQuery(deployment_id="없음")
            )


class Test로그:
    def test_받아_적고_무엇이_들어_있는지_돌려준다(self, wiring) -> None:
        deploy(wiring)
        view = ingest(wiring, 0)

        assert view.total_count == 300
        assert view.distinct_devices == 3
        assert view.labeled_ratio > 0

    def test_없는_배포에는_적을_수_없다(self, wiring) -> None:
        wiring.logs.bind(DeploymentId.of("dep-1"))
        with pytest.raises(DeploymentNotFound):
            IngestInferenceLog(wiring.deployments, wiring.logs).execute(
                IngestInferenceLogCommand(deployment_id="없음", records=records(0))
            )


class Test관측:
    def test_창의_표본_수는_실제_로그_수로_다시_센다(self, wiring) -> None:
        """미리 적어 둔 숫자를 믿으면 로그가 안 올라온 구간을 못 알아챈다."""
        deploy(wiring)
        ingest(wiring, 0, count=42)
        view = observe(wiring, 0)

        assert view.sample_count == 42
        assert "OPS_WINDOW_TOO_SMALL" in {f.code for f in view.findings}

    def test_로그가_없으면_측정하지_않는다(self, wiring) -> None:
        deploy(wiring)
        view = observe(wiring, 0)
        assert view.p95_ms is None
        assert "LOG_EMPTY" in {f.code for f in view.findings}

    def test_세_측정기를_모두_부른다(self, wiring) -> None:
        deploy(wiring)
        ingest(wiring, 0)
        view = observe(wiring, 0)

        assert view.p95_ms is not None
        assert view.prediction_shift is not None
        assert view.max_psi is not None

    def test_드리프트_측정을_끌_수_있다(self, wiring) -> None:
        deploy(wiring)
        ingest(wiring, 0)
        view = observe(wiring, 0, measure_drift=False)
        assert view.max_psi is None

    def test_관측이_없는_배포는_막힌다(self, wiring) -> None:
        deploy(wiring)
        wiring.watches.clear()
        with pytest.raises(HealthWatchNotFound):
            observe(wiring, 0)

    def test_기준_재고정은_현장_분포로_바꾼다(self, wiring) -> None:
        result = deploy(wiring)
        ingest(wiring, 0)
        wiring.mix.counts = {"NORMAL": 290, "OVERLOAD": 9, "FAULT": 1}

        mix = RebaselineWatch(wiring.deployments, wiring.watches, wiring.mix).execute(
            RebaselineCommand(
                deployment_id="dep-1", window=window(0), reason="1일차 안정 구간"
            )
        )
        assert mix["FAULT"] == pytest.approx(0.0033, abs=1e-3)
        watch = wiring.watches.find_by_id(_watch_id(result.watch_id))
        assert watch.baseline_mix == mix


class Test시간선:
    def prepared(self, wiring, psis):  # noqa: ANN001, ANN201
        deploy(wiring)
        for index, psi in enumerate(psis):
            ingest(wiring, index)
            wiring.drift.psi = psi
            observe(wiring, index)
        return wiring

    def test_창이_쌓여야_시간선이_생긴다(self, wiring) -> None:
        self.prepared(wiring, [0.01, 0.01, 5.0])
        view = GetTimeline(wiring.watches).execute(
            GetTimelineQuery(watch_id="watch-dep-1")
        )
        assert view.window_count == 3

    def test_스파이크와_지속을_구분한다(self, wiring) -> None:
        self.prepared(wiring, [0.01, 5.0, 0.01, 3.0, 4.0, 4.5])
        view = FindOnset(wiring.watches).execute(
            FindOnsetQuery(
                watch_id="watch-dep-1",
                metric=HealthMetric.INPUT_PSI,
                threshold=0.2,
                consecutive=3,
            )
        )
        assert view.first_exceeded != view.sustained_from
        assert view.is_sustained

    def test_배포로_관측을_찾는다(self, wiring) -> None:
        self.prepared(wiring, [0.01])
        view = GetWatchForDeployment(wiring.deployments, wiring.watches).execute(
            GetWatchForDeploymentQuery(deployment_id="dep-1")
        )
        assert view.watch_id == "watch-dep-1"


class Test격리와롤백:
    def test_이유를_안_주면_최근_관측에서_찾는다(self, wiring) -> None:
        deploy(wiring)
        ingest(wiring, 0)
        wiring.drift.psi = 8.0
        wiring.drift = StubDriftOutOfRange()
        observe(wiring, 0)

        view = QuarantineDeployment(
            wiring.deployments, wiring.watches, wiring.clock
        ).execute(QuarantineCommand(deployment_id="dep-1"))
        assert view.status == "QUARANTINED"
        assert view.quarantine_reason

    def test_근거가_없으면_멈추지_않는다(self, wiring) -> None:
        deploy(wiring)
        ingest(wiring, 0)
        observe(wiring, 0)

        with pytest.raises(ConflictingRequest):
            QuarantineDeployment(
                wiring.deployments, wiring.watches, wiring.clock
            ).execute(QuarantineCommand(deployment_id="dep-1"))

    def test_롤백은_새_버전을_만든다(self, wiring) -> None:
        from application.operations.deploy_model import (
            ReleaseVersion,
            ReleaseVersionCommand,
        )

        deploy(wiring)
        ReleaseVersion(
            wiring.deployments,
            wiring.optimization_runs,
            wiring.training_runs,
            wiring.clock,
        ).execute(
            ReleaseVersionCommand(
                deployment_id="dep-1",
                optimization_run_id="opt-1",
                training_run_id="run-1",
                released_at="2026-05-21 12:00:00",
            )
        )
        view = RollbackDeployment(wiring.deployments, wiring.clock).execute(
            RollbackCommand(
                deployment_id="dep-1",
                to_version=1,
                reason="v2 문제",
                occurred_at="2026-05-23 09:00:00",
            )
        )
        assert view.current_version == 3
        assert view.rollback_count == 1


class Test재학습:
    def test_로그에서_라벨을_직접_센다(self, wiring) -> None:
        deploy(wiring)
        for index in range(6):
            ingest(wiring, index)
            wiring.drift.psi = 0.01 if index < 3 else 5.0
            observe(wiring, index)

        view = DecideRetraining(wiring.watches, wiring.logs).execute(
            DecideRetrainingCommand(watch_id="watch-dep-1")
        )
        assert view.needed
        assert "INPUT_DRIFT" in view.reasons

    def test_없는_관측은_막힌다(self, wiring) -> None:
        with pytest.raises(HealthWatchNotFound):
            DecideRetraining(wiring.watches, wiring.logs).execute(
                DecideRetrainingCommand(watch_id="없음")
            )


class StubDriftOutOfRange:
    def measure(self, deployment_id, window):  # noqa: ANN001, ANN201
        return DriftReport(
            window=window,
            features=(
                FeatureDrift(
                    field_name="temperature_c",
                    psi=8.0,
                    mean_shift_sigma=3.0,
                    out_of_range_ratio=0.8,
                ),
            ),
        )


def _watch_id(value: str):  # noqa: ANN202
    from domain.operations.identifiers import WatchId

    return WatchId.of(value)
