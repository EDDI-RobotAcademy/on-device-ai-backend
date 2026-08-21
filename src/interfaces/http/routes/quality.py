"""Data Quality API. (모듈 2)

여섯 개의 측정 라우트를 하나로 합치지 않았다.
각 축은 서로 다른 기준(policy)을 받고, 서로 다른 것을 세기 때문이다.
OpenAPI 문서에서 각 축이 무엇을 요구하는지 그대로 보이는 편이 낫다.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from application.data_quality.compare_quality import CompareQualityCommand
from application.data_quality.evaluate_quality_gate import (
    EvaluateQualityGateCommand,
    ReopenAssessmentCommand,
)
from application.data_quality.get_assessment import GetAssessmentQuery
from application.data_quality.measure_balance import MeasureBalanceCommand
from application.data_quality.measure_completeness import MeasureCompletenessCommand
from application.data_quality.measure_label_quality import (
    MeasureLabelQualityCommand,
)
from application.data_quality.measure_noise import MeasureNoiseCommand
from application.data_quality.measure_uniqueness import MeasureUniquenessCommand
from application.data_quality.measure_validity import MeasureValidityCommand
from application.data_quality.record_remediation import RecordRemediationCommand
from application.data_quality.score_quality import ScoreQualityCommand
from application.data_quality.start_quality_assessment import (
    StartQualityAssessmentCommand,
)
from interfaces.http.dependencies.container import quality_container_dependency
from interfaces.http.schemas.quality import (
    AssessmentResponse,
    BalancePolicyRequest,
    CompareQualityRequest,
    CompletenessPolicyRequest,
    DimensionResponse,
    GatePolicyRequest,
    MeasureLabelQualityRequest,
    NoisePolicyRequest,
    QualityComparisonResponse,
    QualityGateResponse,
    QualityScoreResponse,
    RemediationRequest,
    ReopenAssessmentRequest,
    ScoreQualityRequest,
    StartAssessmentRequest,
    UniquenessPolicyRequest,
    ValidityPolicyRequest,
)

router = APIRouter(tags=["data-quality"])


@router.post(
    "/datasets/{dataset_id}/quality-assessments",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_assessment(
    dataset_id: str,
    request: StartAssessmentRequest,
    container: quality_container_dependency,
) -> AssessmentResponse:
    """품질 평가를 시작한다. (실습 2-1)"""
    view = container.start_quality_assessment().execute(
        StartQualityAssessmentCommand(
            assessment_id=request.assessment_id, dataset_id=dataset_id
        )
    )
    return AssessmentResponse.from_view(view)


@router.get(
    "/datasets/{dataset_id}/quality-assessments",
    response_model=list[AssessmentResponse],
)
def list_assessments(
    dataset_id: str, container: quality_container_dependency
) -> list[AssessmentResponse]:
    return [
        AssessmentResponse.from_view(v)
        for v in container.list_assessments().execute(dataset_ref=dataset_id)
    ]


@router.get(
    "/quality-assessments/{assessment_id}", response_model=AssessmentResponse
)
def get_assessment(
    assessment_id: str, container: quality_container_dependency
) -> AssessmentResponse:
    view = container.get_assessment().execute(
        GetAssessmentQuery(assessment_id=assessment_id)
    )
    return AssessmentResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/dimensions/completeness",
    response_model=DimensionResponse,
)
def measure_completeness(
    assessment_id: str,
    request: CompletenessPolicyRequest,
    container: quality_container_dependency,
) -> DimensionResponse:
    """결측 — 값이 없다. (실습 2-2)"""
    view = container.measure_completeness().execute(
        MeasureCompletenessCommand(
            assessment_id=assessment_id, policy=request.to_domain()
        )
    )
    return DimensionResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/dimensions/validity",
    response_model=DimensionResponse,
)
def measure_validity(
    assessment_id: str,
    request: ValidityPolicyRequest,
    container: quality_container_dependency,
) -> DimensionResponse:
    """이상치 — 값이 말이 안 된다. (실습 2-3)"""
    view = container.measure_validity().execute(
        MeasureValidityCommand(assessment_id=assessment_id, policy=request.to_domain())
    )
    return DimensionResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/dimensions/label-quality",
    response_model=DimensionResponse,
)
def measure_label_quality(
    assessment_id: str,
    request: MeasureLabelQualityRequest,
    container: quality_container_dependency,
) -> DimensionResponse:
    """라벨 오류 — 정답이 틀렸다. (실습 2-4)

    규칙 없이 호출하면 '규칙이 없다'는 사실 자체가 CRITICAL 로 돌아온다.
    """
    view = container.measure_label_quality().execute(
        MeasureLabelQualityCommand(
            assessment_id=assessment_id,
            rules=request.to_rules(),
            policy=request.policy.to_domain(),
        )
    )
    return DimensionResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/dimensions/balance",
    response_model=DimensionResponse,
)
def measure_balance(
    assessment_id: str,
    request: BalancePolicyRequest,
    container: quality_container_dependency,
) -> DimensionResponse:
    """불균형 — 한쪽으로 쏠렸다. (실습 2-5)"""
    view = container.measure_balance().execute(
        MeasureBalanceCommand(assessment_id=assessment_id, policy=request.to_domain())
    )
    return DimensionResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/dimensions/noise",
    response_model=DimensionResponse,
)
def measure_noise(
    assessment_id: str,
    request: NoisePolicyRequest,
    container: quality_container_dependency,
) -> DimensionResponse:
    """잡음 — 신호가 묻혔다. (실습 2-6)"""
    view = container.measure_noise().execute(
        MeasureNoiseCommand(assessment_id=assessment_id, policy=request.to_domain())
    )
    return DimensionResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/dimensions/uniqueness",
    response_model=DimensionResponse,
)
def measure_uniqueness(
    assessment_id: str,
    request: UniquenessPolicyRequest,
    container: quality_container_dependency,
) -> DimensionResponse:
    """중복 — 같은 것을 반복해서 가르친다. (실습 2-7)"""
    view = container.measure_uniqueness().execute(
        MeasureUniquenessCommand(
            assessment_id=assessment_id, policy=request.to_domain()
        )
    )
    return DimensionResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/score", response_model=QualityScoreResponse
)
def score_quality(
    assessment_id: str,
    request: ScoreQualityRequest,
    container: quality_container_dependency,
) -> QualityScoreResponse:
    """감이 아니라 숫자로. (실습 2-8)"""
    view = container.score_quality().execute(
        ScoreQualityCommand(
            assessment_id=assessment_id,
            policy=request.policy.to_domain(),
            label_rules=request.to_rules(),
        )
    )
    return QualityScoreResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/remediations",
    response_model=AssessmentResponse,
)
def record_remediation(
    assessment_id: str,
    request: RemediationRequest,
    container: quality_container_dependency,
) -> AssessmentResponse:
    """데이터를 고쳤다는 기록을 남긴다. (실습 2-9)

    기록한 축은 재측정 전까지 게이트를 통과할 수 없다.
    """
    view = container.record_remediation().execute(
        RecordRemediationCommand(
            assessment_id=assessment_id, action=request.to_domain()
        )
    )
    return AssessmentResponse.from_view(view)


@router.post("/quality-comparisons", response_model=QualityComparisonResponse)
def compare_quality(
    request: CompareQualityRequest, container: quality_container_dependency
) -> QualityComparisonResponse:
    """망가진 데이터와 정상 데이터를 직접 비교한다. (실습 2-9)"""
    view = container.compare_quality().execute(
        CompareQualityCommand(
            before_assessment_id=request.before_assessment_id,
            after_assessment_id=request.after_assessment_id,
            before_label=request.before_label,
            after_label=request.after_label,
            policy=request.policy.to_domain(),
        )
    )
    return QualityComparisonResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/gate", response_model=QualityGateResponse
)
def evaluate_gate(
    assessment_id: str,
    request: GatePolicyRequest,
    container: quality_container_dependency,
) -> QualityGateResponse:
    """Data Quality Gate 를 통과시킬지 판정한다. (실습 2-10)"""
    view = container.evaluate_quality_gate().execute(
        EvaluateQualityGateCommand(
            assessment_id=assessment_id, policy=request.to_domain()
        )
    )
    return QualityGateResponse.from_view(view)


@router.post(
    "/quality-assessments/{assessment_id}/reopen", response_model=AssessmentResponse
)
def reopen_assessment(
    assessment_id: str,
    request: ReopenAssessmentRequest,
    container: quality_container_dependency,
) -> AssessmentResponse:
    """판정을 되돌린다. (실습 2-10)"""
    view = container.reopen_assessment().execute(
        ReopenAssessmentCommand(assessment_id=assessment_id, reason=request.reason)
    )
    return AssessmentResponse.from_view(view)
