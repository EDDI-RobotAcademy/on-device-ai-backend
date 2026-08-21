"""Dataset API.

라우트는 얇다. (CLAUDE.md §9)
    요청 검증 → Use Case 호출 → 응답 매핑
그 이상은 하지 않는다. if 문으로 합격 여부를 계산하는 코드가 여기 생기면 설계가 실패한 것이다.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from application.data.analyze_representativeness import (
    AnalyzeRepresentativenessCommand,
)
from application.data.certify_dataset_readiness import (
    CertifyDatasetReadinessCommand,
    ReopenDatasetCommand,
)
from application.data.declare_data_schema import DeclareDataSchemaCommand
from application.data.define_label_space import DefineLabelSpaceCommand
from application.data.design_training_data import DesignTrainingDataCommand
from application.data.get_dataset import GetDatasetQuery
from application.data.infer_data_schema import InferDataSchemaCommand
from application.data.inspect_signal_plausibility import (
    InspectSignalPlausibilityCommand,
)
from application.data.inspect_time_axis import InspectTimeAxisCommand
from application.data.partition_dataset import PartitionDatasetCommand
from application.data.profile_dataset import ProfileDatasetCommand
from application.data.register_dataset import RegisterDatasetCommand
from application.data.support import load_dataset
from application.shared.errors import UnsupportedOperation
from interfaces.http.dependencies.container import container_dependency
from interfaces.http.schemas.dataset import (
    DatasetProfileResponse,
    DatasetResponse,
    DeclareSchemaRequest,
    DefineLabelSpaceRequest,
    DesignTrainingDataRequest,
    InspectionResponse,
    PartitionRequest,
    PartitionResponse,
    ReadinessRequest,
    ReadinessResponse,
    RegisterDatasetRequest,
    ReopenRequest,
    RepresentativenessRequest,
    RepresentativenessResponse,
    SchemaDraftResponse,
    SignalPolicyRequest,
    TimeAxisPolicyRequest,
    TrainingDesignResponse,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def register_dataset(
    request: RegisterDatasetRequest, container: container_dependency
) -> DatasetResponse:
    """현장 데이터를 시스템에 등록한다. (실습 1-1)"""
    view = container.register_dataset().execute(
        RegisterDatasetCommand(
            dataset_id=request.dataset_id,
            name=request.name,
            uri=request.uri,
            source_format=request.to_format(),
            modality=request.to_modality(),
            collected_from=request.collected_from,
        )
    )
    return DatasetResponse.from_view(view)


@router.get("", response_model=list[DatasetResponse])
def list_datasets(container: container_dependency) -> list[DatasetResponse]:
    return [
        DatasetResponse.from_view(v) for v in container.list_datasets().execute()
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, container: container_dependency) -> DatasetResponse:
    view = container.get_dataset().execute(GetDatasetQuery(dataset_id=dataset_id))
    return DatasetResponse.from_view(view)


@router.post("/{dataset_id}/profile", response_model=DatasetProfileResponse)
def profile_dataset(
    dataset_id: str, container: container_dependency
) -> DatasetProfileResponse:
    """데이터를 열어 사실을 확보한다. (실습 1-1)"""
    view = container.profile_dataset().execute(
        ProfileDatasetCommand(dataset_id=dataset_id)
    )
    return DatasetProfileResponse.from_view(view)


@router.get("/{dataset_id}/schema/draft", response_model=SchemaDraftResponse)
def infer_schema(
    dataset_id: str, container: container_dependency
) -> SchemaDraftResponse:
    """스키마 초안을 받는다. 확정은 사람이 한다. (실습 1-3)"""
    view = container.infer_data_schema().execute(
        InferDataSchemaCommand(dataset_id=dataset_id)
    )
    return SchemaDraftResponse(
        dataset_id=view.dataset_id,
        fields=[
            {"name": name, "type": type_name, "role": role}
            for name, type_name, role in view.fields
        ],
        undecided_fields=list(view.undecided_fields),
    )


@router.put("/{dataset_id}/schema", response_model=InspectionResponse)
def declare_schema(
    dataset_id: str, request: DeclareSchemaRequest, container: container_dependency
) -> InspectionResponse:
    """스키마를 확정하고 현실과 대조한다. (실습 1-2, 1-3)"""
    view = container.declare_data_schema().execute(
        DeclareDataSchemaCommand(dataset_id=dataset_id, schema=request.to_domain())
    )
    return InspectionResponse.from_view(view)


@router.post("/{dataset_id}/inspections/signal", response_model=InspectionResponse)
def inspect_signal(
    dataset_id: str, request: SignalPolicyRequest, container: container_dependency
) -> InspectionResponse:
    """센서/이미지가 물리적으로 말이 되는지 본다. (실습 1-4)"""
    view = container.inspect_signal_plausibility().execute(
        InspectSignalPlausibilityCommand(
            dataset_id=dataset_id, policy=request.to_domain()
        )
    )
    return InspectionResponse.from_view(view)


@router.post("/{dataset_id}/inspections/time-axis", response_model=InspectionResponse)
def inspect_time_axis(
    dataset_id: str, request: TimeAxisPolicyRequest, container: container_dependency
) -> InspectionResponse:
    """시간축을 검증한다. (실습 1-5)"""
    view = container.inspect_time_axis().execute(
        InspectTimeAxisCommand(dataset_id=dataset_id, policy=request.to_domain())
    )
    return InspectionResponse.from_view(view)


@router.put("/{dataset_id}/label-space", response_model=InspectionResponse)
def define_label_space(
    dataset_id: str, request: DefineLabelSpaceRequest, container: container_dependency
) -> InspectionResponse:
    """정상과 이상의 정의를 확정한다. (실습 1-6)"""
    view = container.define_label_space().execute(
        DefineLabelSpaceCommand(
            dataset_id=dataset_id,
            label_space=request.to_domain(),
            policy=request.policy.to_domain(),
        )
    )
    return InspectionResponse.from_view(view)


@router.put("/{dataset_id}/training-spec", response_model=TrainingDesignResponse)
def design_training_data(
    dataset_id: str,
    request: DesignTrainingDataRequest,
    container: container_dependency,
) -> TrainingDesignResponse:
    """모델 입력 계약을 확정한다. (실습 1-7)"""
    dataset = load_dataset(container.repository, dataset_id)
    if dataset.schema is None:
        raise UnsupportedOperation(
            "스키마가 없다. 먼저 스키마를 선언한다.", subject=dataset_id
        )
    view = container.design_training_data().execute(
        DesignTrainingDataCommand(
            dataset_id=dataset_id,
            spec=request.to_domain(dataset.schema),
            fit_normalization=request.fit_normalization,
        )
    )
    return TrainingDesignResponse.from_view(view)


@router.post("/{dataset_id}/partitions", response_model=PartitionResponse)
def partition_dataset(
    dataset_id: str, request: PartitionRequest, container: container_dependency
) -> PartitionResponse:
    """계획대로 나누고 누수를 확인한다. (실습 1-8)"""
    view = container.partition_dataset().execute(
        PartitionDatasetCommand(
            dataset_id=dataset_id,
            plan=request.to_plan(),
            policy=request.to_policy(),
        )
    )
    return PartitionResponse.from_view(view)


@router.post(
    "/{dataset_id}/inspections/representativeness",
    response_model=RepresentativenessResponse,
)
def analyze_representativeness(
    dataset_id: str,
    request: RepresentativenessRequest,
    container: container_dependency,
) -> RepresentativenessResponse:
    """학습 데이터가 현실을 대표하는지 본다. (실습 1-9)"""
    dataset = load_dataset(container.repository, dataset_id)
    view = container.analyze_representativeness().execute(
        AnalyzeRepresentativenessCommand(
            dataset_id=dataset_id,
            observed=request.to_source(dataset.source.modality),
            policy=request.to_policy(),
        )
    )
    return RepresentativenessResponse.from_view(view)


@router.post("/{dataset_id}/readiness", response_model=ReadinessResponse)
def certify_readiness(
    dataset_id: str, request: ReadinessRequest, container: container_dependency
) -> ReadinessResponse:
    """학습을 시작해도 되는지 판정한다. (실습 1-10)"""
    view = container.certify_dataset_readiness().execute(
        CertifyDatasetReadinessCommand(
            dataset_id=dataset_id, policy=request.to_policy()
        )
    )
    return ReadinessResponse.from_view(view)


@router.post("/{dataset_id}/reopen", response_model=DatasetResponse)
def reopen_dataset(
    dataset_id: str, request: ReopenRequest, container: container_dependency
) -> DatasetResponse:
    """판정을 되돌린다. (실습 1-10)"""
    container.reopen_dataset().execute(
        ReopenDatasetCommand(dataset_id=dataset_id, reason=request.reason)
    )
    return DatasetResponse.from_view(
        container.get_dataset().execute(GetDatasetQuery(dataset_id=dataset_id))
    )
