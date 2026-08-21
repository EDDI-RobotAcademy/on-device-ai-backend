"""확장 실습 API. (실습 1-11, 2-11, 3-11~3-15, 4-11~4-14, 6-12~6-14)

기존 라우터와 같은 규칙을 지킨다.

    HTTP Request → Validation → Application Use Case → Response Mapping

**판단은 전부 Domain 에 있다.** 여기에는 if 문이 거의 없다.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.exceptions import RequestValidationError

from application.data.design_sampling import DesignSamplingCommand
from application.data_quality.compare_rebalancing import CompareRebalancingCommand
from application.fleet.govern_storage import (
    DeployEndpointCommand,
    GovernStorageCommand,
    InspectStorageCommand,
    RecordExperimentCommand,
    ReviewExperimentCommand,
    TeardownEndpointCommand,
)
from application.model.compare_experiments import (
    CompareExperimentsCommand,
    TrialRequest,
)
from application.model.compare_with_baseline import CompareWithBaselineCommand
from application.model.execute_image_training_run import (
    ExecuteImageTrainingRunCommand,
)
from application.model.prepare_image_training_run import (
    PrepareImageTrainingRunCommand,
)
from application.optimization.compare_quantization import CompareQuantizationCommand
from application.optimization.measure_resources import (
    MeasureResourcesCommand,
    ScaleBatchCommand,
)
from application.optimization.reduce_structure import ReduceStructureCommand
from domain.data.sampling_design import SamplingDesignPolicy, SamplingPlan
from domain.data_quality.rebalancing import RebalancingPlan, RebalancingStrategy
from domain.fleet.endpoint import (
    EndpointPolicy,
    EndpointSpec,
    EndpointVariant,
    OnlineInferenceProfile,
)
from domain.fleet.experiment_record import ExperimentRecord
from domain.fleet.governance import AccessStatement
from domain.model.architecture import (
    ArchitectureKind,
    GlobalPooling,
    ModelArchitecture,
)
from domain.model.experiment import ExperimentPolicy
from domain.model.image_data_ref import ImageReadinessPolicy
from domain.model.statistical_baseline import (
    BaselineJustificationPolicy,
    DetectionMethod,
    DetectorSpec,
)
from domain.model.tensor_spec import ImageTensorSpec
from domain.model.training_config import TrainingConfig
from domain.optimization.benchmark import MeasurementProtocol
from domain.optimization.resource import ResourceBudget, ResourcePolicy
from domain.optimization.structural import (
    ReductionKind,
    StructuralPolicy,
    StructuralReduction,
)
from interfaces.http.dependencies.container import (
    container_dependency,
    fleet_container_dependency,
    model_container_dependency,
    optimization_container_dependency,
    quality_container_dependency,
)
from interfaces.http.schemas.extended import (
    BaselineComparisonResponse,
    BatchScalingResponse,
    BucketGovernanceResponse,
    CompareExperimentsRequest,
    CompareQuantizationRequest,
    CompareRebalancingRequest,
    CompareWithBaselineRequest,
    DeployEndpointRequest,
    DesignSamplingRequest,
    EndpointResponse,
    ExperimentBoardResponse,
    ExperimentLedgerResponse,
    GovernStorageRequest,
    MeasureResourcesRequest,
    PrepareImageTrainingRunRequest,
    QuantizationComparisonResponse,
    RebalancingComparisonResponse,
    RecordExperimentRequest,
    ReduceStructureRequest,
    ReductionComparisonResponse,
    ResourceUsageResponse,
    SamplingTradeoffResponse,
    ScaleBatchRequest,
)
from interfaces.http.schemas.model import PreparationResponse, TrainingCurveResponse

def _enum(enum_type, name: str, label: str):  # noqa: ANN001, ANN201
    """모르는 이름은 **경계에서** 422 로 바꾼다. Domain 까지 들여보내지 않는다."""
    try:
        return enum_type[name]
    except KeyError as exc:
        raise RequestValidationError(
            [
                {
                    "type": "enum",
                    "loc": ("body", label),
                    "msg": f"{label} '{name}' 는 없다. "
                    f"가능한 값: {[m.name for m in enum_type]}",
                    "input": name,
                }
            ]
        ) from exc


data_router = APIRouter(prefix="/datasets", tags=["data (확장)"])
model_router = APIRouter(prefix="/models", tags=["model (확장)"])
optimization_router = APIRouter(
    prefix="/optimization-runs", tags=["optimization (확장)"]
)
experiment_router = APIRouter(prefix="/experiments", tags=["experiment tracking"])
storage_router = APIRouter(prefix="/storage", tags=["storage governance"])
endpoint_router = APIRouter(prefix="/endpoints", tags=["online inference"])


# ---------------------------------------------------------------------------
# 1-11
# ---------------------------------------------------------------------------
@data_router.post("/{dataset_id}/sampling-design", response_model=SamplingTradeoffResponse)
def design_sampling(
    dataset_id: str, request: DesignSamplingRequest, container: container_dependency
) -> SamplingTradeoffResponse:
    view = container.design_sampling().execute(
        DesignSamplingCommand(
            dataset_id=dataset_id,
            plans=tuple(SamplingPlan(**p.model_dump()) for p in request.plans),
            normal_label=request.normal_label,
            value_field=request.value_field,
            policy=SamplingDesignPolicy(
                target_event_seconds=request.target_event_seconds,
                min_samples_per_event=request.min_samples_per_event,
            ),
        )
    )
    return SamplingTradeoffResponse.from_view(view)


# ---------------------------------------------------------------------------
# 2-11
# ---------------------------------------------------------------------------
@data_router.post(
    "/{dataset_id}/rebalancing-comparison", response_model=RebalancingComparisonResponse
)
def compare_rebalancing(
    dataset_id: str,
    request: CompareRebalancingRequest,
    container: quality_container_dependency,
) -> RebalancingComparisonResponse:
    view = container.compare_rebalancing().execute(
        CompareRebalancingCommand(
            dataset_id=dataset_id,
            plans=tuple(
                RebalancingPlan(
                    strategy=_enum(RebalancingStrategy, p.strategy, "전략"),
                    target_ratio=p.target_ratio,
                    applied_after_split=p.applied_after_split,
                )
                for p in request.plans
            ),
        )
    )
    return RebalancingComparisonResponse.from_view(view)


# ---------------------------------------------------------------------------
# 3-11
# ---------------------------------------------------------------------------
@model_router.post(
    "/image-training-runs",
    response_model=PreparationResponse,
    status_code=status.HTTP_201_CREATED,
)
def prepare_image_training_run(
    request: PrepareImageTrainingRunRequest, container: model_container_dependency
) -> PreparationResponse:
    spec = ImageTensorSpec(
        width=request.spec.width,
        height=request.spec.height,
        channels=request.spec.channels,
    )
    architecture = ModelArchitecture(
        kind=ArchitectureKind.CNN2D,
        input_spec=spec.to_tensor_spec(),
        class_count=request.class_count,
        hidden_channels=tuple(request.hidden_channels),
        kernel_size=request.kernel_size,
        pooling=_enum(GlobalPooling, request.pooling, '풀링'),
    )
    view = container.prepare_image_training_run().execute(
        PrepareImageTrainingRunCommand(
            run_id=request.run_id,
            dataset_ref=request.dataset_ref,
            root_uri=request.root_uri,
            spec=spec,
            architecture=architecture,
            config=TrainingConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                seed=request.seed,
            ),
            readiness_policy=ImageReadinessPolicy(),
            require_gates=request.require_gates,
        )
    )
    return PreparationResponse.from_view(view)


@model_router.post(
    "/image-training-runs/{run_id}/execution",
    response_model=TrainingCurveResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def execute_image_training_run(
    run_id: str, container: model_container_dependency
) -> TrainingCurveResponse:
    view = container.execute_image_training_run().execute(
        ExecuteImageTrainingRunCommand(run_id=run_id)
    )
    return TrainingCurveResponse.from_view(view)


# ---------------------------------------------------------------------------
# 3-12 / 3-14 / 3-15
# ---------------------------------------------------------------------------
@model_router.post("/experiments/comparison", response_model=ExperimentBoardResponse)
def compare_experiments(
    request: CompareExperimentsRequest, container: model_container_dependency
) -> ExperimentBoardResponse:
    view = container.compare_experiments().execute(
        CompareExperimentsCommand(
            name=request.name,
            trials=tuple(
                TrialRequest(
                    run_id=t.run_id, label=t.label, knobs=dict(t.knobs), split=t.split
                )
                for t in request.trials
            ),
            metric=request.metric,
            policy=ExperimentPolicy(
                noise_band=request.noise_band,
                min_evaluated_samples=request.min_evaluated_samples,
            ),
        )
    )
    return ExperimentBoardResponse.from_view(view)


# ---------------------------------------------------------------------------
# 3-13
# ---------------------------------------------------------------------------
@model_router.post(
    "/{run_id}/baseline-comparison", response_model=BaselineComparisonResponse
)
def compare_with_baseline(
    run_id: str,
    request: CompareWithBaselineRequest,
    container: model_container_dependency,
) -> BaselineComparisonResponse:
    view = container.compare_with_baseline().execute(
        CompareWithBaselineCommand(
            run_id=run_id,
            detector=DetectorSpec(
                method=_enum(DetectionMethod, request.method, '검출 방법'),
                threshold=request.threshold,
                min_flagged_ratio=request.min_flagged_ratio,
            ),
            normal_label=request.normal_label,
            split=request.split,
            policy=BaselineJustificationPolicy(
                min_recall_gain=request.min_recall_gain
            ),
        )
    )
    return BaselineComparisonResponse.from_view(view)


# ---------------------------------------------------------------------------
# 4-11 ~ 4-14
# ---------------------------------------------------------------------------
@optimization_router.post(
    "/{run_id}/structural-reduction",
    response_model=ReductionComparisonResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def reduce_structure(
    run_id: str,
    request: ReduceStructureRequest,
    container: optimization_container_dependency,
) -> ReductionComparisonResponse:
    view = container.reduce_structure().execute(
        ReduceStructureCommand(
            run_id=run_id,
            reductions=tuple(
                (
                    r.label,
                    StructuralReduction(
                        kind=_enum(ReductionKind, r.kind, '축소 방식'),
                        ratio=r.ratio,
                        fine_tuned=r.fine_tuned,
                    ),
                )
                for r in request.reductions
            ),
            split=request.split,
            fine_tune_epochs=request.fine_tune_epochs,
            policy=StructuralPolicy(max_accuracy_drop=request.max_accuracy_drop),
        )
    )
    return ReductionComparisonResponse.from_view(view)


@optimization_router.post(
    "/{run_id}/quantization-comparison",
    response_model=QuantizationComparisonResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def compare_quantization(
    run_id: str,
    request: CompareQuantizationRequest,
    container: optimization_container_dependency,
) -> QuantizationComparisonResponse:
    view = container.compare_quantization().execute(
        CompareQuantizationCommand(
            run_id=run_id,
            bits=request.bits,
            split=request.split,
            epochs=request.epochs,
            per_channel=request.per_channel,
        )
    )
    return QuantizationComparisonResponse.from_view(view)


@optimization_router.post(
    "/{run_id}/resource-usage", response_model=list[ResourceUsageResponse]
)
def measure_resources(
    run_id: str,
    request: MeasureResourcesRequest,
    container: optimization_container_dependency,
) -> list[ResourceUsageResponse]:
    views = container.measure_resources().execute(
        MeasureResourcesCommand(
            run_id=run_id,
            labels=tuple(request.labels),
            protocol=MeasurementProtocol(
                warmup_runs=request.warmup_runs,
                measured_runs=request.measured_runs,
                batch_size=request.batch_size,
                threads=request.threads,
            ),
            policy=ResourcePolicy(
                budget=ResourceBudget(
                    max_rss_bytes=request.max_rss_bytes, max_cores=request.max_cores
                )
            ),
        )
    )
    return [ResourceUsageResponse.from_view(v) for v in views]


@optimization_router.post(
    "/{run_id}/batch-scaling", response_model=BatchScalingResponse
)
def scale_batch(
    run_id: str,
    request: ScaleBatchRequest,
    container: optimization_container_dependency,
) -> BatchScalingResponse:
    view = container.scale_batch().execute(
        ScaleBatchCommand(
            run_id=run_id,
            label=request.label,
            batch_sizes=tuple(request.batch_sizes),
            protocol=MeasurementProtocol(
                warmup_runs=request.warmup_runs, measured_runs=request.measured_runs
            ),
            cycle_time_ms=request.cycle_time_ms,
        )
    )
    return BatchScalingResponse.from_view(view)


# ---------------------------------------------------------------------------
# 6-12
# ---------------------------------------------------------------------------
@experiment_router.post(
    "/{experiment_id}/trials", status_code=status.HTTP_201_CREATED
)
def record_experiment(
    experiment_id: str,
    request: RecordExperimentRequest,
    container: fleet_container_dependency,
) -> dict[str, list[str]]:
    keys = container.record_experiment().execute(
        RecordExperimentCommand(
            records=tuple(
                ExperimentRecord(experiment_id=experiment_id, **r.model_dump())
                for r in request.records
            )
        )
    )
    return {"keys": list(keys)}


@experiment_router.get("/{experiment_id}", response_model=ExperimentLedgerResponse)
def review_experiment(
    experiment_id: str,
    container: fleet_container_dependency,
    metric: str = "macro_f1",
) -> ExperimentLedgerResponse:
    view = container.review_experiment().execute(
        ReviewExperimentCommand(experiment_id=experiment_id, metric=metric)
    )
    return ExperimentLedgerResponse.from_view(view)


# ---------------------------------------------------------------------------
# 6-13
# ---------------------------------------------------------------------------
@storage_router.put("/governance", response_model=BucketGovernanceResponse)
def govern_storage(
    request: GovernStorageRequest, container: fleet_container_dependency
) -> BucketGovernanceResponse:
    view = container.govern_storage().execute(
        GovernStorageCommand(
            versioning=request.versioning,
            encryption=request.encryption,
            block_public_access=request.block_public_access,
            expiration_days=request.expiration_days,
            statements=tuple(
                AccessStatement(
                    sid=s.sid,
                    effect=s.effect,
                    principal=s.principal,
                    actions=tuple(s.actions),
                    resources=tuple(s.resources),
                )
                for s in request.statements
            ),
            version_prefix=request.version_prefix,
        )
    )
    return BucketGovernanceResponse.from_view(view)


@storage_router.get("/governance", response_model=BucketGovernanceResponse)
def inspect_storage(
    container: fleet_container_dependency, version_prefix: str = ""
) -> BucketGovernanceResponse:
    view = container.inspect_storage().execute(
        InspectStorageCommand(version_prefix=version_prefix)
    )
    return BucketGovernanceResponse.from_view(view)


# ---------------------------------------------------------------------------
# 6-14
# ---------------------------------------------------------------------------
@endpoint_router.post(
    "", response_model=EndpointResponse, status_code=status.HTTP_201_CREATED
)
def deploy_endpoint(
    request: DeployEndpointRequest, container: fleet_container_dependency
) -> EndpointResponse:
    view = container.deploy_endpoint().execute(
        DeployEndpointCommand(
            spec=EndpointSpec(
                name=request.name,
                variants=tuple(
                    EndpointVariant(**v.model_dump()) for v in request.variants
                ),
            ),
            profile=OnlineInferenceProfile(
                cycle_time_ms=request.cycle_time_ms,
                network_round_trip_ms=request.network_round_trip_ms,
                inference_ms=request.inference_ms,
                offline_tolerance_minutes=request.offline_tolerance_minutes,
                requests_per_hour=request.requests_per_hour,
            ),
            image_uri=request.image_uri,
            policy=EndpointPolicy(),
        )
    )
    return EndpointResponse.from_view(view)


@endpoint_router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def teardown_endpoint(name: str, container: fleet_container_dependency) -> None:
    """**실습이 끝나면 반드시 부른다.** 켜 두면 시간당 과금된다 (실습 6-14)."""
    container.teardown_endpoint().execute(TeardownEndpointCommand(name=name))
