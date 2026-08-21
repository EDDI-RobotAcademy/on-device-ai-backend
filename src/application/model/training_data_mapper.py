"""Data / Data Quality → Model 번역기 (Anti-Corruption Layer).

Model Context 는 Dataset 도 QualityAssessment 도 모른다.
그런데 **두 게이트를 통과했다는 사실**은 알아야 한다. 그것이 학습 시작의 전제이기 때문이다.

그래서 이 번역기가 세 Context 를 잇는다.

    Dataset (모듈 1)           → uri, 입력 필드, 라벨, 정규화 통계, READY 여부
    QualityAssessment (모듈 2) → Data Quality Gate 통과 여부
                               ↓
    TrainingDataRef            → 원시 값만 담긴 VO
"""

from __future__ import annotations

from application.shared.errors import UnsupportedOperation
from domain.data.dataset import Dataset, DatasetStatus
from domain.data_quality.assessment import AssessmentStatus, QualityAssessment
from domain.model.training_data_ref import TrainingDataRef


def training_data_from(
    dataset: Dataset, assessment: QualityAssessment | None = None
) -> TrainingDataRef:
    """학습에 쓸 데이터 참조를 만든다.

    게이트를 통과하지 않았어도 **번역 자체는 된다.**
    막는 것은 TrainingRun.prepare() 의 일이다 — 판단은 Domain 이 한다.
    """
    schema = dataset.schema
    spec = dataset.training_spec
    if schema is None or spec is None:
        raise UnsupportedOperation(
            "학습 설계(실습 1-7)가 확정되지 않은 Dataset 으로는 학습할 수 없다.",
            subject=str(dataset.id),
        )

    return TrainingDataRef(
        dataset_ref=str(dataset.id),
        uri=dataset.source.uri,
        source_format=dataset.source.format.value,
        feature_fields=spec.feature_fields,
        label_field=spec.label_field,
        time_field=schema.time_index.name if schema.time_index else None,
        normalization=dict(spec.normalization.statistics),
        split_ratio=_split_ratio(dataset),
        readiness_certified=dataset.status is DatasetStatus.READY,
        quality_gate_passed=(
            assessment is not None
            and assessment.status is AssessmentStatus.PASSED
            and assessment.dataset_ref == str(dataset.id)
        ),
    )


def _split_ratio(dataset: Dataset) -> tuple[float, float, float]:
    partition = dataset.partition
    if partition is None:
        return (0.7, 0.15, 0.15)
    ratio = partition.plan.ratio
    return (ratio.train, ratio.validation, ratio.test)
