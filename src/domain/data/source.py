"""데이터의 출처(Value Object).

실습 1-1 / 1-2 의 뼈대.

현장 데이터는 "파일"이 아니라 "어느 설비가, 어떤 방식으로, 어떤 주기로 뱉은 신호"다.
경로(uri)만 들고 있는 구조를 만들면 그 맥락이 사라진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation


class Modality(Enum):
    """데이터가 세상을 기록하는 방식."""

    TIME_SERIES = "TIME_SERIES"
    """센서가 시간축을 따라 남긴 값. 순서와 간격이 의미를 가진다."""

    IMAGE = "IMAGE"
    """카메라가 한 순간을 남긴 격자. 조명·초점·해상도가 의미를 가진다."""

    TABULAR = "TABULAR"
    """시간축이 없는 표. 행 사이의 순서에 의미가 없다."""


class SourceFormat(Enum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    IMAGE_DIRECTORY = "IMAGE_DIRECTORY"


_FORMAT_MODALITIES: dict[SourceFormat, frozenset[Modality]] = {
    SourceFormat.CSV: frozenset({Modality.TIME_SERIES, Modality.TABULAR}),
    SourceFormat.PARQUET: frozenset({Modality.TIME_SERIES, Modality.TABULAR}),
    SourceFormat.IMAGE_DIRECTORY: frozenset({Modality.IMAGE}),
}


@dataclass(frozen=True, slots=True)
class DataSourceDescriptor:
    """데이터가 어디서 어떻게 왔는지에 대한 서술.

    Domain 은 이 값으로 파일을 열지 않는다. 여는 것은 Infrastructure 의 일이다.
    """

    uri: str
    format: SourceFormat
    modality: Modality
    collected_from: str
    """현장 식별자. 예: "LINE-3 / PRESS-07", "DIECAST-CELL-A"."""

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise InvariantViolation("데이터 출처 uri 는 비어 있을 수 없다.", subject="uri")
        if not self.collected_from.strip():
            raise InvariantViolation(
                "어느 현장에서 수집했는지 없는 데이터는 추적이 불가능하다.",
                subject="collected_from",
            )
        allowed = _FORMAT_MODALITIES[self.format]
        if self.modality not in allowed:
            raise InvariantViolation(
                f"{self.format.value} 형식은 {self.modality.value} 를 담을 수 없다.",
                subject="modality",
            )

    @property
    def is_time_series(self) -> bool:
        return self.modality is Modality.TIME_SERIES

    @property
    def is_image(self) -> bool:
        return self.modality is Modality.IMAGE
