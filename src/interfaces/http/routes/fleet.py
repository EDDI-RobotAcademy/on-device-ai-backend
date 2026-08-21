"""Fleet API. (모듈 6)

    /fleets      수천 대를 어떻게 아는가
    /rollouts    새 모델을 어떻게 내보내는가

**이 파일에도 AWS 가 없다.** 클라이언트는 데이터가 어디로 가는지 알 필요가 없다.

오래 걸리는 것은 둘이다 — 원격 학습(6-5)과 OTA(6-8).
둘 다 Command → Job → Status → Result 로 되어 있다 (CLAUDE.md §11).
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, status

from application.fleet.build_and_train import (
    BuildDatasetCommand,
    PollTrainingQuery,
    SubmitTrainingCommand,
)
from application.fleet.ingest_uplink import (
    CreateFleetCommand,
    IngestUplinkCommand,
    InspectLakeCommand,
    MarkDeviceCommand,
    SummarizeFleetQuery,
    SweepStaleCommand,
)
from application.fleet.release_and_rollout import (
    AdvanceRolloutCommand,
    ApplyOutcomesCommand,
    CollectWaveCommand,
    GetRolloutQuery,
    HaltRolloutCommand,
    PlanRolloutCommand,
    PromoteReleaseCommand,
    PublishReleaseCommand,
    RollbackRolloutCommand,
    TraceLineageQuery,
)
from interfaces.http.dependencies.container import fleet_container_dependency
from interfaces.http.schemas.fleet import (
    AdvanceRolloutRequest,
    ApplyOutcomesRequest,
    BuildDatasetRequest,
    ChannelResponse,
    CollectWaveRequest,
    CreateFleetRequest,
    DatasetBuildResponse,
    FleetResponse,
    HaltRolloutRequest,
    IngestUplinkRequest,
    InspectLakeRequest,
    LakeLayoutResponse,
    LineageResponse,
    MarkDeviceRequest,
    PlanRolloutRequest,
    PromoteReleaseRequest,
    PublishReleaseRequest,
    ReleaseResponse,
    RollbackRolloutRequest,
    RolloutResponse,
    SubmitTrainingRequest,
    SweepRequest,
    TraceLineageRequest,
    TrainingJobResponse,
    UplinkResponse,
    WaveResponse,
)

router = APIRouter(prefix="/fleets", tags=["fleet"])
rollout_router = APIRouter(prefix="/rollouts", tags=["fleet"])
training_router = APIRouter(prefix="/cloud-training-jobs", tags=["fleet"])


# ---------------------------------------------------------------------------
# 플릿과 데이터
# ---------------------------------------------------------------------------
@router.post("", response_model=FleetResponse, status_code=status.HTTP_201_CREATED)
def create_fleet(
    request: CreateFleetRequest, container: fleet_container_dependency
) -> FleetResponse:
    """플릿을 만들고 디바이스를 등록한다. (실습 6-1, 6-3)"""
    view = container.create_fleet().execute(
        CreateFleetCommand(
            fleet_id=request.fleet_id,
            name=request.name,
            devices=tuple(d.to_domain() for d in request.devices),
        )
    )
    return FleetResponse.from_view(view)


@router.post(
    "/{fleet_id}/uplinks",
    response_model=UplinkResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_uplink(
    fleet_id: str, request: IngestUplinkRequest, container: fleet_container_dependency
) -> UplinkResponse:
    """디바이스가 묶음 하나를 올린다. (실습 6-1, 6-2)

    **거절도 결과다.** 개인정보가 섞였으면 저장하지 않고 그 사실을 돌려준다.
    """
    view = container.ingest_uplink().execute(
        IngestUplinkCommand(
            fleet_id=fleet_id,
            batch=request.batch.to_domain(),
            body=base64.b64decode(request.body_base64 or ""),
            layout=request.layout.to_domain(),
            policy=request.policy.to_domain(),
            part=request.part,
        )
    )
    return UplinkResponse.from_view(view)


@router.post("/{fleet_id}/lake", response_model=LakeLayoutResponse)
def inspect_lake(
    fleet_id: str, request: InspectLakeRequest, container: fleet_container_dependency
) -> LakeLayoutResponse:
    """이 키 설계로 나중에 살 수 있는가. (실습 6-2)"""
    view = container.inspect_lake_layout().execute(
        InspectLakeCommand(
            fleet_id=fleet_id,
            layout=request.layout.to_domain(),
            policy=request.policy.to_domain(),
            filters=request.filters,
        )
    )
    return LakeLayoutResponse.from_view(view)


@router.get("/{fleet_id}", response_model=FleetResponse)
def summarize_fleet(
    fleet_id: str, container: fleet_container_dependency
) -> FleetResponse:
    """3,000대를 여섯 줄로. (실습 6-3, 6-11)"""
    view = container.summarize_fleet().execute(
        SummarizeFleetQuery(fleet_id=fleet_id)
    )
    return FleetResponse.from_view(view)


@router.post("/{fleet_id}/devices/{device_id}/status", response_model=FleetResponse)
def mark_device(
    fleet_id: str,
    device_id: str,
    request: MarkDeviceRequest,
    container: fleet_container_dependency,
) -> FleetResponse:
    """격리·정비·폐기. 모듈 5 의 격리가 여기로 들어온다."""
    view = container.mark_device().execute(
        MarkDeviceCommand(
            fleet_id=fleet_id,
            device_id=device_id,
            status=request.status_target(),
            note=request.note,
        )
    )
    return FleetResponse.from_view(view)


@router.post("/{fleet_id}/sweep", response_model=FleetResponse)
def sweep_stale(
    fleet_id: str, request: SweepRequest, container: fleet_container_dependency
) -> FleetResponse:
    """오래 연락 없는 디바이스를 표시한다. (실습 6-11)"""
    view = container.sweep_stale_devices().execute(
        SweepStaleCommand(
            fleet_id=fleet_id,
            now=request.now,
            stale_after=request.stale_after,
            unreachable_after=request.unreachable_after,
        )
    )
    return FleetResponse.from_view(view)


@router.post(
    "/{fleet_id}/datasets",
    response_model=DatasetBuildResponse,
    status_code=status.HTTP_201_CREATED,
)
def build_dataset(
    fleet_id: str, request: BuildDatasetRequest, container: fleet_container_dependency
) -> DatasetBuildResponse:
    """현장 데이터로 학습 데이터셋을 만든다. (실습 6-4)

    **격리된 디바이스는 자동으로 빠지고, 뺐다는 사실이 기록에 남는다.**
    """
    view = container.build_training_dataset().execute(
        BuildDatasetCommand(
            fleet_id=fleet_id,
            build_id=request.build_id,
            window=request.window.to_domain(),
            record_counts=request.record_counts,
            labeled_counts=request.labeled_counts,
            label_distribution=request.label_distribution,
            include_devices=tuple(request.include_devices),
            policy=request.policy.to_domain(),
        )
    )
    return DatasetBuildResponse.from_view(view)


@router.post(
    "/{fleet_id}/releases",
    response_model=ReleaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def publish_release(
    fleet_id: str, request: PublishReleaseRequest, container: fleet_container_dependency
) -> ReleaseResponse:
    """디바이스로 내보낼 수 있게 묶는다. (실습 6-6)"""
    view = container.publish_release().execute(
        PublishReleaseCommand(
            fleet_id=fleet_id,
            bundle=request.bundle.to_domain(),
            policy=request.policy.to_domain(),
        )
    )
    return ReleaseResponse.from_view(view)


@router.post("/{fleet_id}/channels", response_model=ChannelResponse)
def promote_release(
    fleet_id: str, request: PromoteReleaseRequest, container: fleet_container_dependency
) -> ChannelResponse:
    """채널에 올린다. (실습 6-7)

    **한 채널에 하나뿐이고, STABLE 로 가려면 CANARY 를 거쳐야 한다.**
    """
    channels = container.promote_release().execute(
        PromoteReleaseCommand(
            fleet_id=fleet_id,
            version=request.version,
            channel=request.channel_target(),
        )
    )
    return ChannelResponse(fleet_id=fleet_id, channels=channels)


@router.post(
    "/{fleet_id}/rollouts",
    response_model=RolloutResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def plan_rollout(
    fleet_id: str, request: PlanRolloutRequest, container: fleet_container_dependency
) -> RolloutResponse:
    """단계를 짜고 첫 단계를 시작한다. (실습 6-8)

    **202 다.** 알린 것이지 도착한 것이 아니다 — 며칠 걸린다.
    """
    view = container.plan_rollout().execute(
        PlanRolloutCommand(
            fleet_id=fleet_id,
            rollout_id=request.rollout_id,
            version=request.version,
            wave_sizes=tuple(request.wave_sizes),
            group_order=tuple(request.group_order),
            policy=request.policy.to_domain(),
            occurred_at=request.occurred_at,
        )
    )
    return RolloutResponse.from_view(view)


@router.post(
    "/{fleet_id}/devices/{device_id}/lineage", response_model=LineageResponse
)
def trace_lineage(
    fleet_id: str,
    device_id: str,
    request: TraceLineageRequest,
    container: fleet_container_dependency,
) -> LineageResponse:
    """이 디바이스의 모델이 어느 데이터에서 왔는가. (실습 6-10)"""
    view = container.trace_lineage().execute(
        TraceLineageQuery(
            fleet_id=fleet_id,
            device_id=device_id,
            source_devices=tuple(request.source_devices),
            window=request.window,
        )
    )
    return LineageResponse.from_view(view)


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
@training_router.post(
    "",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_training(
    request: SubmitTrainingRequest, container: fleet_container_dependency
) -> TrainingJobResponse:
    """클라우드에 학습을 맡긴다. (실습 6-5)

    **제출은 즉시 끝난다.** 예산을 넘으면 제출 자체를 안 한다.
    """
    view = container.submit_training_job().execute(
        SubmitTrainingCommand(
            job_id=request.job_id,
            dataset_uri=request.dataset_uri,
            output_uri=request.output_uri,
            compute=request.compute.to_domain(),
            hyperparameters=request.hyperparameters,
            policy=request.policy.to_domain(),
        )
    )
    return TrainingJobResponse.from_view(view)


@training_router.get("/{job_id}", response_model=TrainingJobResponse)
def poll_training(
    job_id: str, container: fleet_container_dependency
) -> TrainingJobResponse:
    """지금 어떻게 됐는지 물어본다. **기다리지 않는다.**"""
    view = container.poll_training_job().execute(PollTrainingQuery(job_id=job_id))
    return TrainingJobResponse.from_view(view)


# ---------------------------------------------------------------------------
# 롤아웃
# ---------------------------------------------------------------------------
@rollout_router.get("", response_model=list[RolloutResponse])
def list_rollouts(container: fleet_container_dependency) -> list[RolloutResponse]:
    return [RolloutResponse.from_view(v) for v in container.list_rollouts().execute()]


@rollout_router.get("/{rollout_id}", response_model=RolloutResponse)
def get_rollout(
    rollout_id: str, container: fleet_container_dependency
) -> RolloutResponse:
    view = container.get_rollout().execute(GetRolloutQuery(rollout_id=rollout_id))
    return RolloutResponse.from_view(view)


@rollout_router.post("/{rollout_id}/collect", response_model=WaveResponse)
def collect_wave(
    rollout_id: str, request: CollectWaveRequest, container: fleet_container_dependency
) -> WaveResponse:
    """디바이스들이 뭐라고 했는지 걷어 적는다. (실습 6-8)

    **응답이 없는 것은 PENDING 이다.** 실패로 세지 않는다.
    """
    view = container.collect_wave().execute(
        CollectWaveCommand(
            rollout_id=rollout_id,
            policy=request.policy.to_domain(),
            occurred_at=request.occurred_at,
        )
    )
    return WaveResponse.from_view(view)


@rollout_router.post("/{rollout_id}/advance", response_model=RolloutResponse)
def advance_rollout(
    rollout_id: str,
    request: AdvanceRolloutRequest,
    container: fleet_container_dependency,
) -> RolloutResponse:
    """다음 단계로 넘어간다. (실습 6-8)"""
    view = container.advance_rollout().execute(
        AdvanceRolloutCommand(
            rollout_id=rollout_id,
            fleet_id=request.fleet_id,
            policy=request.policy.to_domain(),
            occurred_at=request.occurred_at,
        )
    )
    return RolloutResponse.from_view(view)


@rollout_router.post("/{rollout_id}/halt", response_model=RolloutResponse)
def halt_rollout(
    rollout_id: str, request: HaltRolloutRequest, container: fleet_container_dependency
) -> RolloutResponse:
    view = container.halt_rollout().execute(
        HaltRolloutCommand(
            rollout_id=rollout_id,
            reason=request.reason,
            occurred_at=request.occurred_at,
        )
    )
    return RolloutResponse.from_view(view)


@rollout_router.post(
    "/{rollout_id}/rollback",
    response_model=RolloutResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def rollback_rollout(
    rollout_id: str,
    request: RollbackRolloutRequest,
    container: fleet_container_dependency,
) -> RolloutResponse:
    """되돌린다. **그런데 즉시 안 된다** — 202 다. (실습 6-9)"""
    view = container.rollback_rollout().execute(
        RollbackRolloutCommand(
            fleet_id=request.fleet_id,
            rollout_id=rollout_id,
            new_rollout_id=request.new_rollout_id,
            reason=request.reason,
            to_version=request.to_version,
            wave_sizes=tuple(request.wave_sizes),
            policy=request.policy.to_domain(),
            occurred_at=request.occurred_at,
        )
    )
    return RolloutResponse.from_view(view)


@rollout_router.post("/{rollout_id}/apply", response_model=RolloutResponse)
def apply_outcomes(
    rollout_id: str,
    request: ApplyOutcomesRequest,
    container: fleet_container_dependency,
) -> RolloutResponse:
    """디바이스가 보고한 버전을 플릿에 반영한다.

    **서버가 '보냈으니 됐겠지'라고 적지 않는다.**
    """
    view = container.apply_rollout_outcomes().execute(
        ApplyOutcomesCommand(
            fleet_id=request.fleet_id,
            rollout_id=rollout_id,
            seen_at=request.seen_at,
        )
    )
    return RolloutResponse.from_view(view)
