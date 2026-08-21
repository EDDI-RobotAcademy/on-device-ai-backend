"""Operations API. (모듈 5)

두 자원으로 나뉜다. **Aggregate 가 둘이기 때문이다.**

    /deployments      무엇이 어디에 올라가 있는가 — 결정
    /health-watches   그것이 아직 괜찮은가        — 사실

배포·격리·롤백은 즉시 끝난다. 파일을 옮기는 것이 아니라 **기록을 남기는 일**이다.
오래 걸리는 것은 로그 수집과 관측이고, 그것은 디바이스가 보내오는 쪽이다.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from application.operations.compare_shadow import CompareShadowCommand
from application.operations.decide_retraining import DecideRetrainingCommand
from application.operations.deploy_model import (
    DeployModelCommand,
    GetDeploymentQuery,
    ReleaseVersionCommand,
)
from application.operations.find_onset import (
    FindOnsetQuery,
    GetTimelineQuery,
    GetWatchForDeploymentQuery,
)
from application.operations.observe_health import (
    IngestInferenceLogCommand,
    ObserveHealthCommand,
    RebaselineCommand,
)
from application.operations.respond_to_incident import (
    ListIncidentsQuery,
    QuarantineCommand,
    ResolveIncidentCommand,
    ResumeCommand,
    RollbackCommand,
)
from interfaces.http.dependencies.container import operations_container_dependency
from interfaces.http.schemas.operations import (
    BaselineResponse,
    CompareShadowRequest,
    DecideRetrainingRequest,
    DeployModelRequest,
    DeploymentResponse,
    DeployResponse,
    FindOnsetRequest,
    HealthReportResponse,
    IncidentResponse,
    IngestInferenceLogRequest,
    LogCoverageResponse,
    ObserveHealthRequest,
    OnsetResponse,
    QuarantineRequest,
    RebaselineRequest,
    ReleaseCheckResponse,
    ReleaseVersionRequest,
    ResolveIncidentRequest,
    ResumeRequest,
    RetrainingResponse,
    RollbackRequest,
    ShadowResponse,
    TimelineResponse,
    WatchResponse,
)

router = APIRouter(prefix="/deployments", tags=["operations"])
watch_router = APIRouter(prefix="/health-watches", tags=["operations"])


# ---------------------------------------------------------------------------
# 배포
# ---------------------------------------------------------------------------
@router.post("", response_model=DeployResponse, status_code=status.HTTP_201_CREATED)
def deploy_model(
    request: DeployModelRequest, container: operations_container_dependency
) -> DeployResponse:
    """모델을 현장에 배포한다. (실습 5-1)

    **관측(HealthWatch)이 함께 열린다.** 나중에 켜면 그 사이 구간은 영영 비어 있다.
    """
    result = container.deploy_model().execute(
        DeployModelCommand(
            deployment_id=request.deployment_id,
            optimization_run_id=request.optimization_run_id,
            target=request.target.to_domain(),
            training_run_id=request.training_run_id,
            artifact_label=request.artifact_label,
            watch_id=request.watch_id,
            note=request.note,
            released_at=request.released_at,
            require_selected=request.require_selected,
            policy=request.policy.to_domain(),
        )
    )
    return DeployResponse(
        deployment=DeploymentResponse.from_view(result.deployment),
        check=ReleaseCheckResponse.from_view(result.check),
        watch_id=result.watch_id,
    )


@router.get("", response_model=list[DeploymentResponse])
def list_deployments(
    container: operations_container_dependency,
) -> list[DeploymentResponse]:
    return [
        DeploymentResponse.from_view(v) for v in container.list_deployments().execute()
    ]


@router.post(
    "/{deployment_id}/versions",
    response_model=DeployResponse,
    status_code=status.HTTP_201_CREATED,
)
def release_version(
    deployment_id: str,
    request: ReleaseVersionRequest,
    container: operations_container_dependency,
) -> DeployResponse:
    """새 버전을 올린다. (실습 5-2)"""
    result = container.release_version().execute(
        ReleaseVersionCommand(
            deployment_id=deployment_id,
            optimization_run_id=request.optimization_run_id,
            training_run_id=request.training_run_id,
            artifact_label=request.artifact_label,
            note=request.note,
            released_at=request.released_at,
            require_selected=request.require_selected,
            policy=request.policy.to_domain(),
        )
    )
    return DeployResponse(
        deployment=DeploymentResponse.from_view(result.deployment),
        check=ReleaseCheckResponse.from_view(result.check),
    )


@router.post(
    "/{deployment_id}/logs",
    response_model=LogCoverageResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_inference_log(
    deployment_id: str,
    request: IngestInferenceLogRequest,
    container: operations_container_dependency,
) -> LogCoverageResponse:
    """디바이스가 판단 기록을 올린다. (실습 5-3)

    응답에 **이 로그로 답할 수 있는 질문이 있는지**가 함께 온다.
    "받았습니다"만 돌려주면 반년 뒤에야 로그가 쓸모없다는 걸 알게 된다.
    """
    from domain.operations.identifiers import DeploymentId

    container.logs.bind(DeploymentId.of(deployment_id))
    view = container.ingest_inference_log().execute(
        IngestInferenceLogCommand(
            deployment_id=deployment_id,
            records=tuple(r.to_domain() for r in request.records),
            policy=request.policy.to_domain(),
        )
    )
    return LogCoverageResponse.from_view(view)


@router.post("/{deployment_id}/observations", response_model=HealthReportResponse)
def observe_health(
    deployment_id: str,
    request: ObserveHealthRequest,
    container: operations_container_dependency,
) -> HealthReportResponse:
    """창 하나를 본다. (실습 5-4 ~ 5-7)

    지연시간·예측분포·입력분포를 **한 번에** 본다. 따로 보면 원인을 못 짚는다.
    """
    view = container.observe_health().execute(
        ObserveHealthCommand(
            deployment_id=deployment_id,
            window=request.window.to_domain(),
            latency_policy=request.latency_policy.to_domain(),
            mix_policy=request.mix_policy.to_domain(),
            drift_policy=request.drift_policy.to_domain(),
            log_policy=request.log_policy.to_domain(),
            window_policy=request.window_policy.to_domain(),
            incident_policy=request.incident_policy.to_domain(),
            open_incident=request.open_incident,
            measure_drift=request.measure_drift,
        )
    )
    return HealthReportResponse.from_view(view)


@router.post("/{deployment_id}/baseline", response_model=BaselineResponse)
def rebaseline(
    deployment_id: str,
    request: RebaselineRequest,
    container: operations_container_dependency,
) -> BaselineResponse:
    """기준 예측 분포를 현장 안정 구간으로 다시 잡는다. (실습 5-6)

    **판정 기준을 바꾸는 일이다.** 그래서 이유가 필수다.
    """
    mix = container.rebaseline_watch().execute(
        RebaselineCommand(
            deployment_id=deployment_id,
            window=request.window.to_domain(),
            reason=request.reason,
        )
    )
    return BaselineResponse(deployment_id=deployment_id, baseline_mix=mix)


@router.post("/{deployment_id}/quarantine", response_model=DeploymentResponse)
def quarantine(
    deployment_id: str,
    request: QuarantineRequest,
    container: operations_container_dependency,
) -> DeploymentResponse:
    """판단을 멈춘다. (실습 5-8)

    이유를 안 주면 최근 관측에서 근거를 찾아 붙인다.
    근거가 없으면 409 — 이유 없이 현장 판단을 멈추지 않는다.
    """
    view = container.quarantine_deployment().execute(
        QuarantineCommand(
            deployment_id=deployment_id,
            reason=request.reason,
            occurred_at=request.occurred_at,
            policy=request.policy.to_domain(),
        )
    )
    return DeploymentResponse.from_view(view)


@router.post("/{deployment_id}/resume", response_model=DeploymentResponse)
def resume(
    deployment_id: str,
    request: ResumeRequest,
    container: operations_container_dependency,
) -> DeploymentResponse:
    view = container.resume_deployment().execute(
        ResumeCommand(
            deployment_id=deployment_id,
            reason=request.reason,
            occurred_at=request.occurred_at,
        )
    )
    return DeploymentResponse.from_view(view)


@router.post("/{deployment_id}/rollback", response_model=DeploymentResponse)
def rollback(
    deployment_id: str,
    request: RollbackRequest,
    container: operations_container_dependency,
) -> DeploymentResponse:
    """이전 버전으로 되돌린다. (실습 5-10)

    **버전 번호는 줄어들지 않는다.** v3 → v1 로 돌아가면 결과는 v4 다.
    """
    view = container.rollback_deployment().execute(
        RollbackCommand(
            deployment_id=deployment_id,
            to_version=request.to_version,
            reason=request.reason,
            occurred_at=request.occurred_at,
        )
    )
    return DeploymentResponse.from_view(view)


@router.post("/{deployment_id}/shadow", response_model=ShadowResponse)
def compare_shadow(
    deployment_id: str,
    request: CompareShadowRequest,
    container: operations_container_dependency,
) -> ShadowResponse:
    """새 모델을 같은 입력에 나란히 돌려 본다. (실습 5-9)"""
    view = container.compare_shadow().execute(
        CompareShadowCommand(
            deployment_id=deployment_id,
            window=request.window.to_domain(),
            candidate_artifact_id=request.candidate_artifact_id,
            policy=request.policy.to_domain(),
        )
    )
    return ShadowResponse.from_view(view)


@router.get("/{deployment_id}/watch", response_model=WatchResponse)
def get_watch_for_deployment(
    deployment_id: str, container: operations_container_dependency
) -> WatchResponse:
    view = container.get_watch_for_deployment().execute(
        GetWatchForDeploymentQuery(deployment_id=deployment_id)
    )
    return WatchResponse.from_view(view)


@router.get("/{deployment_id}", response_model=DeploymentResponse)
def get_deployment(
    deployment_id: str, container: operations_container_dependency
) -> DeploymentResponse:
    view = container.get_deployment().execute(
        GetDeploymentQuery(deployment_id=deployment_id)
    )
    return DeploymentResponse.from_view(view)


# ---------------------------------------------------------------------------
# 관측
# ---------------------------------------------------------------------------
@watch_router.get("", response_model=list[WatchResponse])
def list_watches(container: operations_container_dependency) -> list[WatchResponse]:
    return [WatchResponse.from_view(v) for v in container.list_watches().execute()]


@watch_router.get("/{watch_id}", response_model=WatchResponse)
def get_watch(
    watch_id: str, container: operations_container_dependency
) -> WatchResponse:
    view = container.get_watch().execute(GetTimelineQuery(watch_id=watch_id))
    return WatchResponse.from_view(view)


@watch_router.get("/{watch_id}/timeline", response_model=TimelineResponse)
def get_timeline(
    watch_id: str, container: operations_container_dependency
) -> TimelineResponse:
    """창들을 시간 순으로 늘어놓는다. (실습 5-4)"""
    view = container.get_timeline().execute(GetTimelineQuery(watch_id=watch_id))
    return TimelineResponse.from_view(view)


@watch_router.post("/{watch_id}/onset", response_model=OnsetResponse)
def find_onset(
    watch_id: str,
    request: FindOnsetRequest,
    container: operations_container_dependency,
) -> OnsetResponse:
    """언제부터 이상해졌는지 찾는다. (실습 5-4)

    **한 번 튄 것과 무너진 것을 구분해서** 돌려준다.
    """
    view = container.find_onset().execute(
        FindOnsetQuery(
            watch_id=watch_id,
            metric=request.metric_target(),
            threshold=request.threshold,
            consecutive=request.consecutive,
        )
    )
    return OnsetResponse.from_view(view)


@watch_router.get("/{watch_id}/incidents", response_model=list[IncidentResponse])
def list_incidents(
    watch_id: str,
    container: operations_container_dependency,
    only_open: bool = False,
) -> list[IncidentResponse]:
    return [
        IncidentResponse.from_view(v)
        for v in container.list_incidents().execute(
            ListIncidentsQuery(watch_id=watch_id, only_open=only_open)
        )
    ]


@watch_router.post(
    "/{watch_id}/incidents/{incident_id}/resolution", response_model=IncidentResponse
)
def resolve_incident(
    watch_id: str,
    incident_id: str,
    request: ResolveIncidentRequest,
    container: operations_container_dependency,
) -> IncidentResponse:
    view = container.resolve_incident().execute(
        ResolveIncidentCommand(
            watch_id=watch_id,
            incident_id=incident_id,
            resolution=request.resolution,
        )
    )
    return IncidentResponse.from_view(view)


@watch_router.post("/{watch_id}/retraining", response_model=RetrainingResponse)
def decide_retraining(
    watch_id: str,
    request: DecideRetrainingRequest,
    container: operations_container_dependency,
) -> RetrainingResponse:
    """재학습이 필요한가. (실습 5-11)

    "필요하다/아니다"가 아니라 **"지금 시작할 수 있는가, 무엇이 막고 있는가"** 를 돌려준다.
    """
    view = container.decide_retraining().execute(
        DecideRetrainingCommand(
            watch_id=watch_id,
            supply=request.supply.to_domain() if request.supply else None,
            measured_accuracy=request.measured_accuracy,
            policy=request.policy.to_domain(),
        )
    )
    return RetrainingResponse.from_view(view)
