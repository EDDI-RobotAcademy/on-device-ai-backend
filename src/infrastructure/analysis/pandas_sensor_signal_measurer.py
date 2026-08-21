"""SensorSignalMeasurer 구현 (pandas/numpy). — 실습 1-4

세는 것은 세 가지다.
    1. 물리 범위를 벗어난 표본 수      ← 스키마의 ValueRange 가 기준을 준다
    2. 값이 변하지 않고 연속된 최대 길이 ← 센서 고착
    3. 관측 최대/최소값에 붙어 있는 표본 수 ← 포화(clipping)
"""

from __future__ import annotations

from itertools import groupby

import numpy as np
import pandas as pd

from domain.data.schema import DataSchema, FieldRole
from domain.data.signal import SensorChannelMeasurement
from domain.data.source import DataSourceDescriptor
from infrastructure.analysis.table_loader import load_table, numeric_view


class PandasSensorSignalMeasurer:
    """domain.data.ports.SensorSignalMeasurer 구현."""

    def measure(
        self, source: DataSourceDescriptor, schema: DataSchema
    ) -> tuple[SensorChannelMeasurement, ...]:
        frame = load_table(source).frame

        measurements: list[SensorChannelMeasurement] = []
        for spec in schema.fields:
            if spec.role is not FieldRole.FEATURE or not spec.type.is_numeric:
                continue
            if spec.name not in frame.columns:
                continue

            values = numeric_view(frame[spec.name])
            present = values.dropna()
            total = int(len(present))
            if total == 0:
                measurements.append(
                    SensorChannelMeasurement(field_name=spec.name, total_count=0)
                )
                continue

            out_of_range = 0
            if spec.value_range is not None:
                out_of_range = int(
                    (
                        (present < spec.value_range.minimum)
                        | (present > spec.value_range.maximum)
                    ).sum()
                )

            measurements.append(
                SensorChannelMeasurement(
                    field_name=spec.name,
                    total_count=total,
                    out_of_range_count=out_of_range,
                    longest_constant_run=_longest_constant_run(values),
                    saturated_count=_saturated_count(present),
                )
            )

        return tuple(measurements)


def _longest_constant_run(values: pd.Series) -> int:
    """값이 한 번도 변하지 않고 이어진 최대 길이.

    NaN 은 연속을 끊는다. 값이 없는 것과 값이 고정된 것은 다른 사건이다.
    """
    array = values.to_numpy(dtype="float64", na_value=np.nan)
    longest = 0
    for value, group in groupby(array):
        if np.isnan(value):
            continue
        longest = max(longest, sum(1 for _ in group))
    return int(longest)


def _saturated_count(present: pd.Series) -> int:
    """관측 최대/최소값에 정확히 붙어 있는 표본 수.

    정상 신호라면 최대값은 한 번만 나온다.
    여러 번 나온다면 그 지점에서 값이 잘려 나갔다는 뜻이다.
    """
    if present.empty:
        return 0
    at_max = int((present == present.max()).sum())
    at_min = int((present == present.min()).sum())
    # 정상적으로 한 번씩 등장하는 극값은 포화가 아니다.
    saturated = max(at_max - 1, 0) + max(at_min - 1, 0)
    return min(saturated, int(len(present)))
