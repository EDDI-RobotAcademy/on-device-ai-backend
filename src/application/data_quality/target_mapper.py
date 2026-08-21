"""Data Context → Data Quality Context 번역기 (Anti-Corruption Layer).

두 Context 는 서로를 모른다. 그 사이를 잇는 것이 Application 의 일이다.

여기서 하는 일은 단 하나: `Dataset` 이라는 남의 Aggregate 를
`AssessmentTarget` 이라는 우리 쪽 언어로 바꾼다.

이 파일이 있어서, Data Context 의 스키마 구조가 바뀌어도
Data Quality Context 의 코드는 한 줄도 바뀌지 않는다.
"""

from __future__ import annotations

from domain.data.dataset import Dataset
from domain.data.schema import FieldRole
from domain.data_quality.target import AssessmentTarget
from application.shared.errors import UnsupportedOperation


def assessment_target_from(dataset: Dataset) -> AssessmentTarget:
    """학습 설계가 확정된 Dataset 을 품질 평가 대상으로 번역한다.

    입력 필드는 스키마의 FEATURE 가 아니라 **학습 설계(TrainingDataSpec)** 에서 가져온다.
    품질은 "모델이 실제로 보게 될 열"에 대해 따져야 의미가 있기 때문이다.
    """
    schema = dataset.schema
    if schema is None:
        raise UnsupportedOperation(
            "스키마가 없다. 어떤 열을 볼지 모르면 품질을 평가할 수 없다.",
            subject=str(dataset.id),
        )

    spec = dataset.training_spec
    if spec is not None:
        feature_fields = spec.feature_fields
        label_field = spec.label_field
    else:
        feature_fields = tuple(f.name for f in schema.feature_fields)
        label_field = schema.label_field.name if schema.label_field else None

    if not feature_fields:
        raise UnsupportedOperation(
            "입력 필드가 하나도 없다.", subject=str(dataset.id)
        )

    group_fields = schema.fields_with_role(FieldRole.GROUP)
    physical_ranges = {
        f.name: (f.value_range.minimum, f.value_range.maximum)
        for f in schema.fields
        if f.value_range is not None
    }

    return AssessmentTarget(
        dataset_ref=str(dataset.id),
        uri=dataset.source.uri,
        source_format=dataset.source.format.value,
        feature_fields=feature_fields,
        label_field=label_field,
        time_field=schema.time_index.name if schema.time_index else None,
        group_field=group_fields[0].name if group_fields else None,
        physical_ranges=physical_ranges,
    )
