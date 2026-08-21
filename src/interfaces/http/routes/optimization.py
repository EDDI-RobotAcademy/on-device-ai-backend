"""Optimization API. (모듈 4)

변환도 오래 걸리는 작업이다 (CLAUDE.md §11).
TFLite INT8 변환은 대표 데이터를 전부 흘려 보내야 하고, 벤치마크는 수백 번을 돈다.

    POST /optimization-runs                     → 201  OPEN
    POST /optimization-runs/{id}/baseline       → 202  기준 측정 (백그라운드)
    POST /optimization-runs/{id}/candidates     → 202  변환      (백그라운드)
    GET  /optimization-runs/{id}                →      진행 상황을 물어본다
    GET  /optimization-runs/{id}/tradeoff       →      비교표
    POST /optimization-runs/{id}/selection      → 200  선택은 즉시 — 계산이 아니라 판단이다

Route 는 얇다. 판단은 전부 Domain 에 있다.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, status

from application.optimization.benchmark_baseline import BenchmarkBaselineCommand
from application.optimization.compare_candidates import (
    CompareCandidatesCommand,
    InspectArtifactSizesCommand,
)
from application.optimization.convert_model import ConvertModelCommand
from application.optimization.get_optimization_run import GetOptimizationRunQuery
from application.optimization.profile_roofline import ProfileRooflineCommand
from application.optimization.select_model import (
    ReopenOptimizationRunCommand,
    SelectModelCommand,
)
from application.optimization.start_optimization_run import (
    StartOptimizationRunCommand,
)
from interfaces.http.dependencies.container import optimization_container_dependency
from interfaces.http.schemas.optimization import (
    BenchmarkBaselineRequest,
    BenchmarkResponse,
    CandidateResponse,
    ConvertModelRequest,
    OptimizationRunResponse,
    ProfileRooflineRequest,
    ReopenOptimizationRunRequest,
    RooflineResponse,
    SelectModelRequest,
    SelectionResponse,
    StartOptimizationRunRequest,
    TradeoffResponse,
)

router = APIRouter(prefix="/optimization-runs", tags=["optimization"])


@router.post(
    "", response_model=OptimizationRunResponse, status_code=status.HTTP_201_CREATED
)
def start_optimization_run(
    request: StartOptimizationRunRequest,
    container: optimization_container_dependency,
) -> OptimizationRunResponse:
    """최적화를 시작한다. (실습 4-1)

    승인받지 않은 모델이면 Domain 이 여기서 막는다.
    """
    view = container.start_optimization_run().execute(
        StartOptimizationRunCommand(
            run_id=request.run_id,
            training_run_id=request.training_run_id,
            split=request.split,
            require_accepted=request.require_accepted,
        )
    )
    return OptimizationRunResponse.from_view(view)


@router.get("", response_model=list[OptimizationRunResponse])
def list_optimization_runs(
    container: optimization_container_dependency,
) -> list[OptimizationRunResponse]:
    return [
        OptimizationRunResponse.from_view(v)
        for v in container.list_optimization_runs().execute()
    ]


@router.post(
    "/{run_id}/baseline",
    response_model=BenchmarkResponse,
)
def benchmark_baseline(
    run_id: str,
    request: BenchmarkBaselineRequest,
    container: optimization_container_dependency,
) -> BenchmarkResponse:
    """기준 모델을 잰다. (실습 4-1)

    이 응답에는 숫자와 함께 **프로토콜**이 들어 있다.
    어떻게 쟀는지 없이 'p95 0.03ms'만 돌려주면 그 숫자는 재현할 수 없다.
    """
    view = container.benchmark_baseline().execute(
        BenchmarkBaselineCommand(
            run_id=run_id,
            protocol=request.protocol.to_domain(),
            policy=request.policy.to_domain(),
            split=request.split,
        )
    )
    return BenchmarkResponse.from_view(view)


@router.post(
    "/{run_id}/candidates",
    response_model=CandidateResponse,
    status_code=status.HTTP_201_CREATED,
)
def convert_model(
    run_id: str,
    request: ConvertModelRequest,
    container: optimization_container_dependency,
) -> CandidateResponse:
    """실행 경로/정밀도를 바꿔 후보를 하나 만든다. (실습 4-2 ~ 4-7)

    변환만 하지 않는다. 대조하고, 재고, 다시 평가한 뒤에야 후보가 된다.
    """
    view = container.convert_model().execute(
        ConvertModelCommand(
            run_id=run_id,
            runtime=request.runtime_target(),
            precision=request.precision_target(),
            equivalence_samples=request.equivalence_samples,
            split=request.split,
            protocol=request.protocol.to_domain(),
            equivalence_policy=request.equivalence_policy.to_domain(),
            benchmark_policy=request.benchmark_policy.to_domain(),
        )
    )
    return CandidateResponse.from_view(view)


@router.post(
    "/{run_id}/candidates:async",
    response_model=OptimizationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def convert_model_async(
    run_id: str,
    request: ConvertModelRequest,
    background: BackgroundTasks,
    container: optimization_container_dependency,
) -> OptimizationRunResponse:
    """변환을 백그라운드로 돌린다. **응답은 즉시 돌아온다.** (CLAUDE.md §11)

    INT8 변환은 대표 데이터를 전부 흘려 보내야 해서 오래 걸린다.
    진행 상황은 GET /optimization-runs/{id} 로 물어본다.
    """
    view = container.get_optimization_run().execute(
        GetOptimizationRunQuery(run_id=run_id)
    )
    background.add_task(
        container.convert_model().execute,
        ConvertModelCommand(
            run_id=run_id,
            runtime=request.runtime_target(),
            precision=request.precision_target(),
            equivalence_samples=request.equivalence_samples,
            split=request.split,
            protocol=request.protocol.to_domain(),
            equivalence_policy=request.equivalence_policy.to_domain(),
            benchmark_policy=request.benchmark_policy.to_domain(),
        ),
    )
    return OptimizationRunResponse.from_view(view)


@router.post("/{run_id}/roofline", response_model=RooflineResponse)
def profile_roofline(
    run_id: str,
    request: ProfileRooflineRequest,
    container: optimization_container_dependency,
) -> RooflineResponse:
    """층별 병목 구조를 분석한다. (실습 4-9)"""
    view = container.profile_roofline().execute(
        ProfileRooflineCommand(
            run_id=run_id,
            device=request.device.to_domain(),
            policy=request.policy.to_domain(),
        )
    )
    return RooflineResponse.from_view(view)


@router.get("/{run_id}/roofline", response_model=RooflineResponse)
def get_roofline(
    run_id: str, container: optimization_container_dependency
) -> RooflineResponse:
    view = container.get_roofline_profile().execute(
        GetOptimizationRunQuery(run_id=run_id)
    )
    return RooflineResponse.from_view(view)


@router.get("/{run_id}/tradeoff", response_model=TradeoffResponse)
def compare_candidates(
    run_id: str, container: optimization_container_dependency
) -> TradeoffResponse:
    """정확도와 Latency를 한 표에 놓는다. (실습 4-8)"""
    view = container.compare_candidates().execute(
        CompareCandidatesCommand(run_id=run_id)
    )
    return TradeoffResponse.from_view(view)


@router.get("/{run_id}/artifacts", response_model=list[CandidateResponse])
def inspect_artifact_sizes(
    run_id: str, container: optimization_container_dependency
) -> list[CandidateResponse]:
    """크기를 '가중치 + 오버헤드'로 쪼개 본다. (실습 4-5)"""
    return [
        CandidateResponse.from_view(v)
        for v in container.inspect_artifact_sizes().execute(
            InspectArtifactSizesCommand(run_id=run_id)
        )
    ]


@router.post("/{run_id}/selection", response_model=SelectionResponse)
def select_model(
    run_id: str,
    request: SelectModelRequest,
    container: optimization_container_dependency,
) -> SelectionResponse:
    """예산 안에서 고른다. (실습 4-10)

    선택은 계산이 아니라 판단이므로 즉시 끝난다 — 이미 잰 숫자만 본다.
    """
    view = container.select_model().execute(
        SelectModelCommand(
            run_id=run_id,
            budget=request.budget.to_domain(),
            objective=request.objective_target(),
            equivalence=request.equivalence.to_domain(),
            require_deployable_runtime=request.require_deployable_runtime,
        )
    )
    return SelectionResponse.from_view(view)


@router.get("/{run_id}/selection", response_model=SelectionResponse)
def get_selection(
    run_id: str, container: optimization_container_dependency
) -> SelectionResponse:
    view = container.get_optimization_certificate().execute(
        GetOptimizationRunQuery(run_id=run_id)
    )
    return SelectionResponse.from_view(view)


@router.post("/{run_id}/reopen", response_model=OptimizationRunResponse)
def reopen_optimization_run(
    run_id: str,
    request: ReopenOptimizationRunRequest,
    container: optimization_container_dependency,
) -> OptimizationRunResponse:
    """판정을 되돌린다. **이유 없이는 되돌릴 수 없다.**"""
    container.reopen_optimization_run().execute(
        ReopenOptimizationRunCommand(run_id=run_id, reason=request.reason)
    )
    view = container.get_optimization_run().execute(
        GetOptimizationRunQuery(run_id=run_id)
    )
    return OptimizationRunResponse.from_view(view)


@router.get("/{run_id}", response_model=OptimizationRunResponse)
def get_optimization_run(
    run_id: str, container: optimization_container_dependency
) -> OptimizationRunResponse:
    view = container.get_optimization_run().execute(
        GetOptimizationRunQuery(run_id=run_id)
    )
    return OptimizationRunResponse.from_view(view)
