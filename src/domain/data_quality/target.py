"""품질 평가의 대상 (Anti-Corruption Layer VO).

Data Quality Context 는 Data Context 의 `Dataset` 을 모른다.
알아야 하는 것은 "어디에 있는 무엇을, 어떤 열을 기준으로 볼 것인가"뿐이다.

그래서 원시 값만 담은 이 VO 로 번역해서 받는다.
번역기는 Application Layer 에 있다. (application/data_quality/target_mapper.py)

이 방식의 이득:
    - 두 Context 가 서로의 스키마 변경에 끌려다니지 않는다
    - 품질 평가를 Dataset 없이도(예: 임시 파일) 수행할 수 있다
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class AssessmentTarget:
    """품질을 평가할 데이터 한 덩어리."""

    dataset_ref: str
    """다른 Context 의 Aggregate 는 식별자 문자열로만 참조한다."""

    uri: str
    source_format: str = "CSV"

    feature_fields: tuple[str, ...] = field(default_factory=tuple)
    label_field: str | None = None
    time_field: str | None = None
    group_field: str | None = None

    physical_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    """필드명 → (최소, 최대). 설비 사양서에서 온 값이다."""

    def __post_init__(self) -> None:
        if not self.dataset_ref.strip():
            raise InvariantViolation(
                "어느 Dataset 에 대한 평가인지 없으면 결과를 되돌릴 수 없다.",
                subject="dataset_ref",
            )
        if not self.uri.strip():
            raise InvariantViolation("평가할 데이터 위치가 없다.", subject="uri")
        if not self.feature_fields:
            raise InvariantViolation(
                "입력 필드가 없다. 모델이 볼 열을 모르면 품질을 논할 수 없다.",
                subject="feature_fields",
            )
        if len(self.feature_fields) != len(set(self.feature_fields)):
            raise InvariantViolation("입력 필드가 중복되었다.", subject="feature_fields")
        for name, (minimum, maximum) in self.physical_ranges.items():
            if minimum > maximum:
                raise InvariantViolation(
                    f"'{name}' 의 물리 범위가 뒤집혀 있다.", subject=name
                )

    def range_of(self, field_name: str) -> tuple[float, float] | None:
        return self.physical_ranges.get(field_name)

    @property
    def has_label(self) -> bool:
        return bool(self.label_field)
