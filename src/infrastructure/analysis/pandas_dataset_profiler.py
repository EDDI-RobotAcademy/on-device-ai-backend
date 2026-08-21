"""DatasetProfiler 구현 (pandas). — 실습 1-1

데이터를 열어 사실만 센다. 좋고 나쁨은 말하지 않는다.
"""

from __future__ import annotations

import pandas as pd

from domain.data.profile import ColumnProfile, DatasetProfile, FieldType
from domain.data.source import DataSourceDescriptor
from infrastructure.analysis.table_loader import (
    infer_field_type,
    load_table,
    numeric_view,
)


class PandasDatasetProfiler:
    """domain.data.ports.DatasetProfiler 구현."""

    def __init__(self, sample_value_count: int = 5) -> None:
        self._sample_value_count = sample_value_count

    def profile(self, source: DataSourceDescriptor) -> DatasetProfile:
        table = load_table(source)
        frame = table.frame
        columns = tuple(
            self._profile_column(frame[name]) for name in frame.columns
        )
        return DatasetProfile(
            row_count=int(len(frame)),
            columns=columns,
            byte_size=table.byte_size,
        )

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        field_type = infer_field_type(series)
        total = int(len(series))
        missing = int(series.isna().sum())

        minimum = maximum = mean = stddev = None
        if field_type.is_numeric:
            numeric = numeric_view(series)
            if numeric.notna().any():
                minimum = float(numeric.min())
                maximum = float(numeric.max())
                mean = float(numeric.mean())
                stddev = float(numeric.std(ddof=0))

        present = series.dropna()
        samples = tuple(
            str(v) for v in present.drop_duplicates().head(self._sample_value_count)
        )

        return ColumnProfile(
            name=str(series.name),
            inferred_type=field_type,
            total_count=total,
            missing_count=missing,
            distinct_count=int(present.nunique()),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            stddev=stddev,
            sample_values=samples,
        )


class HeuristicSchemaInferrer:
    """SchemaInferrer 구현. — 실습 1-3

    이름과 타입만 보고 역할을 찍는다. 정확할 리가 없다. 그것이 요점이다.
    추론기는 "여기가 애매하다"를 알려주는 도구이지, 스키마를 결정하는 주체가 아니다.
    """

    _TIME_HINTS = ("time", "timestamp", "datetime", "date", "ts", "측정시각", "시각")
    _LABEL_HINTS = ("label", "class", "condition", "status", "판정", "라벨", "결과")
    _ID_HINTS = ("id", "uuid", "serial", "번호")
    _GROUP_HINTS = ("batch", "lot", "group", "machine", "line", "device", "설비", "로트")

    def infer(self, profile: DatasetProfile):  # noqa: ANN201 - Port 시그니처 유지
        from domain.data.schema import DataSchema, FieldRole, FieldSpec

        specs: list[FieldSpec] = []
        time_index_taken = False
        label_taken = False

        for column in profile.columns:
            lowered = column.name.lower()
            role = FieldRole.METADATA

            if (
                not time_index_taken
                and column.inferred_type is FieldType.TIMESTAMP
                and any(h in lowered for h in self._TIME_HINTS)
            ):
                role = FieldRole.TIME_INDEX
                time_index_taken = True
            elif (
                not label_taken
                and any(h in lowered for h in self._LABEL_HINTS)
                and column.inferred_type in (FieldType.CATEGORY, FieldType.BOOLEAN)
            ):
                role = FieldRole.LABEL
                label_taken = True
            elif any(h in lowered for h in self._GROUP_HINTS) and column.inferred_type in (
                FieldType.CATEGORY,
                FieldType.TEXT,
            ):
                role = FieldRole.GROUP
            elif any(lowered.endswith(h) or lowered == h for h in self._ID_HINTS):
                role = FieldRole.IDENTIFIER
            elif column.inferred_type.is_numeric and not column.is_constant:
                role = FieldRole.FEATURE

            field_type = column.inferred_type
            if field_type is FieldType.UNKNOWN:
                field_type = FieldType.TEXT
            if role is FieldRole.TIME_INDEX:
                field_type = FieldType.TIMESTAMP

            specs.append(
                FieldSpec(
                    name=column.name,
                    type=field_type,
                    role=role,
                    required=column.missing_count == 0,
                )
            )

        return DataSchema(fields=tuple(specs))
