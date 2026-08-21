"""Model API. (모듈 3)

학습은 오래 걸리는 작업이다 (CLAUDE.md §11).
그래서 **요청이 학습을 붙잡고 기다리지 않는다.**

    POST /training-runs             → 201  PREPARED   (즉시)
    POST /training-runs/{id}/start  → 202  RUNNING    (즉시, 학습은 백그라운드)
    GET  /training-runs/{id}        →      진행 상황을 물어본다
    POST /training-runs/{id}/evaluations
    POST /training-runs/{id}/acceptance
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, status

from application.model.accept_model import (
    AcceptModelCommand,
    ReopenTrainingRunCommand,
)
from application.model.evaluate_model import (
    EvaluateModelCommand,
    EvaluateOnFieldCommand,
)
from application.model.execute_training_run import ExecuteTrainingRunCommand
from application.model.get_training_run import GetTrainingRunQuery
from application.model.prepare_training_run import PrepareTrainingRunCommand
from application.model.support import load_run
from interfaces.http.dependencies.container import model_container_dependency
from interfaces.http.schemas.model import (
    AcceptanceRequest,
    ArchitectureProfileResponse,
    EvaluateRequest,
    EvaluationResponse,
    FieldEvaluateRequest,
    ModelCertificateResponse,
    PrepareTrainingRunRequest,
    PreparationResponse,
    ReopenTrainingRunRequest,
    TrainingCurveResponse,
    TrainingRunResponse,
)

router = APIRouter(prefix="/training-runs", tags=["model"])


@router.post(
    "", response_model=PreparationResponse, status_code=status.HTTP_201_CREATED
)
def prepare_training_run(
    request: PrepareTrainingRunRequest, container: model_container_dependency
) -> PreparationResponse:
    """학습을 준비한다. (실습 3-1, 3-2, 3-4)

    데이터를 텐서로 바꿔 보고, 구조를 조립해 보고, 게이트 통과 여부를 확인한다.
    여기까지 통과해야 학습을 시작할 수 있다.
    """
    view = container.prepare_training_run().execute(
        PrepareTrainingRunCommand(
            run_id=request.run_id,
            dataset_id=request.dataset_id,
            assessment_id=request.assessment_id,
            architecture=request.architecture.to_domain(),
            config=request.config.to_domain(),
            windowing=request.windowing.to_domain(),
            require_gates=request.require_gates,
        )
    )
    return PreparationResponse.from_view(view)


@router.get("", response_model=list[TrainingRunResponse])
def list_training_runs(
    container: model_container_dependency,
) -> list[TrainingRunResponse]:
    return [
        TrainingRunResponse.from_view(v)
        for v in container.list_training_runs().execute()
    ]


@router.post(
    "/{run_id}/start",
    response_model=TrainingRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_training_run(
    run_id: str,
    background: BackgroundTasks,
    container: model_container_dependency,
) -> TrainingRunResponse:
    """학습을 시작한다. **응답은 즉시 돌아온다.** (실습 3-5)

    202 Accepted — "받았고, 하고 있다"는 뜻이다.
    진행 상황은 GET /training-runs/{id} 로 물어본다.
    """
    view = container.get_training_run().execute(GetTrainingRunQuery(run_id=run_id))
    background.add_task(
        container.execute_training_run().execute,
        ExecuteTrainingRunCommand(run_id=run_id),
    )
    return TrainingRunResponse.from_view(view)


@router.get("/{run_id}", response_model=TrainingRunResponse)
def get_training_run(
    run_id: str, container: model_container_dependency
) -> TrainingRunResponse:
    """Job 상태를 물어본다."""
    view = container.get_training_run().execute(GetTrainingRunQuery(run_id=run_id))
    return TrainingRunResponse.from_view(view)


@router.get("/{run_id}/curve", response_model=TrainingCurveResponse)
def get_training_curve(
    run_id: str, container: model_container_dependency
) -> TrainingCurveResponse:
    """학습 곡선. (실습 3-6, 3-7)"""
    view = container.get_training_curve().execute(GetTrainingRunQuery(run_id=run_id))
    return TrainingCurveResponse.from_view(view)


@router.get("/{run_id}/architecture", response_model=ArchitectureProfileResponse)
def get_architecture_profile(
    run_id: str, container: model_container_dependency
) -> ArchitectureProfileResponse:
    """층별 계산량. (실습 3-2)"""
    from application.model.dto import ArchitectureProfileView
    from domain.model.errors import ModelNotTrained

    run = load_run(container.runs, run_id)
    if run.profile is None:
        raise ModelNotTrained("구조 프로파일이 없다.", subject=run_id)
    return ArchitectureProfileResponse.from_view(
        ArchitectureProfileView.of(run_id, run.profile)
    )


@router.post("/{run_id}/evaluations", response_model=EvaluationResponse)
def evaluate_model(
    run_id: str, request: EvaluateRequest, container: model_container_dependency
) -> EvaluationResponse:
    """정확도 뒤를 본다. (실습 3-9)"""
    view = container.evaluate_model().execute(
        EvaluateModelCommand(
            run_id=run_id, split=request.split, policy=request.policy.to_domain()
        )
    )
    return EvaluationResponse.from_view(view)


@router.post("/{run_id}/field-evaluations", response_model=EvaluationResponse)
def evaluate_on_field(
    run_id: str, request: FieldEvaluateRequest, container: model_container_dependency
) -> EvaluationResponse:
    """현장 홀드아웃으로 평가한다. (실습 3-10)"""
    view = container.evaluate_model_on_field().execute(
        EvaluateOnFieldCommand(
            run_id=run_id,
            field_uri=request.field_uri,
            split_name=request.split_name,
            policy=request.policy.to_domain(),
        )
    )
    return EvaluationResponse.from_view(view)


@router.post("/{run_id}/acceptance", response_model=ModelCertificateResponse)
def accept_model(
    run_id: str, request: AcceptanceRequest, container: model_container_dependency
) -> ModelCertificateResponse:
    """현장에 내보낼 수 있는지 판정한다. (실습 3-10)"""
    view = container.accept_model().execute(
        AcceptModelCommand(
            run_id=run_id, split=request.split, policy=request.to_domain()
        )
    )
    return ModelCertificateResponse.from_view(view)


@router.post("/{run_id}/reopen", response_model=TrainingRunResponse)
def reopen_training_run(
    run_id: str,
    request: ReopenTrainingRunRequest,
    container: model_container_dependency,
) -> TrainingRunResponse:
    """판정을 되돌린다. (실습 3-10)"""
    container.reopen_training_run().execute(
        ReopenTrainingRunCommand(run_id=run_id, reason=request.reason)
    )
    return TrainingRunResponse.from_view(
        container.get_training_run().execute(GetTrainingRunQuery(run_id=run_id))
    )
