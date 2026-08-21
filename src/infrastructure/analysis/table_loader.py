"""표 형식 원본을 pandas 로 읽는다.

pandas 가 이 파일 밖으로 새 나가지 않게 하는 것이 목적이다.
언젠가 polars 로 바꾼다면 고치는 곳은 analysis 패키지뿐이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from domain.data.profile import FieldType
from domain.data.source import DataSourceDescriptor, SourceFormat
from infrastructure.errors import SourceUnreadable, UnsupportedSourceFormat

_IMAGE_SUFFIX = re.compile(r"\.(?:png|jpe?g|bmp|tiff?)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LoadedTable:
    frame: pd.DataFrame
    byte_size: int


def load_table(source: DataSourceDescriptor) -> LoadedTable:
    """Data Context 의 DataSourceDescriptor 로 읽는다."""
    return load_frame(source.uri, source.format.value)


def load_frame(uri: str, source_format: str = "CSV") -> LoadedTable:
    """문자열 위치/형식으로 읽는다.

    Data Quality Context 는 DataSourceDescriptor 를 모른다 (AssessmentTarget 만 안다).
    두 Context 가 같은 로더를 쓰되 서로의 타입을 알 필요는 없도록 이 형태를 함께 둔다.
    """
    path = Path(uri)
    if not path.exists():
        raise SourceUnreadable(f"원본을 찾을 수 없다: {uri}", subject=uri)

    try:
        if source_format == SourceFormat.CSV.value:
            # 타입 추론은 우리가 직접 한다. 여기서는 pandas 기본 추론만 받는다.
            frame = pd.read_csv(path)
        elif source_format == SourceFormat.PARQUET.value:
            frame = pd.read_parquet(path)
        else:
            raise UnsupportedSourceFormat(
                f"{source_format} 는 표로 읽을 수 없다.", subject=source_format
            )
    except (UnsupportedSourceFormat, SourceUnreadable):
        raise
    except Exception as exc:  # pragma: no cover - 손상 파일 방어
        raise SourceUnreadable(f"원본을 읽지 못했다: {exc}", subject=uri) from exc

    return LoadedTable(frame=frame, byte_size=path.stat().st_size)


def to_datetime(series: pd.Series) -> pd.Series:
    """문자열 시각을 datetime 으로 바꾼다. 실패한 값은 NaT 로 남긴다."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")


def infer_field_type(series: pd.Series) -> FieldType:
    """열이 스스로 주장하는 타입을 읽는다.

    주의: 이것은 *추론*이다. 계약이 아니다.
    0/1 로 된 라벨은 여기서 INTEGER 로 보인다. 그것이 CATEGORY 라는 것은 현장이 안다.
    """
    if pd.api.types.is_bool_dtype(series):
        return FieldType.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return FieldType.TIMESTAMP
    if pd.api.types.is_integer_dtype(series):
        return FieldType.INTEGER
    if pd.api.types.is_float_dtype(series):
        return FieldType.REAL

    values = series.dropna()
    if values.empty:
        return FieldType.UNKNOWN

    text = values.astype(str)
    if text.str.contains(_IMAGE_SUFFIX, regex=True).mean() > 0.9:
        return FieldType.IMAGE_REF

    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    if float(parsed.notna().mean()) > 0.9:
        return FieldType.TIMESTAMP

    distinct = int(text.nunique())
    if distinct <= 50 and distinct / len(text) < 0.5:
        return FieldType.CATEGORY
    return FieldType.TEXT


def numeric_view(series: pd.Series) -> pd.Series:
    """수치로 다룰 수 있는 형태로 강제 변환한다. 변환 실패는 NaN."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce")
