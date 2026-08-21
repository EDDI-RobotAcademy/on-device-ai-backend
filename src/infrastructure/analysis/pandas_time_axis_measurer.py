"""TimeAxisMeasurer 구현 (pandas). — 실습 1-5

중요: 정렬하지 않고 **파일에 적힌 순서 그대로** 센다.
정렬해 버리면 "행이 뒤섞여 들어왔다"는 사실 자체가 사라진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.data.source import DataSourceDescriptor
from domain.data.time_axis import TimeAxisMeasurement
from infrastructure.analysis.table_loader import load_table, to_datetime
from infrastructure.errors import SourceUnreadable


class PandasTimeAxisMeasurer:
    """domain.data.ports.TimeAxisMeasurer 구현."""

    def __init__(self, gap_multiplier: float = 3.0) -> None:
        self._gap_multiplier = gap_multiplier

    def measure(
        self, source: DataSourceDescriptor, time_field: str
    ) -> TimeAxisMeasurement:
        frame = load_table(source).frame
        if time_field not in frame.columns:
            raise SourceUnreadable(
                f"'{time_field}' 열이 원본에 없다.", subject=time_field
            )

        stamps = to_datetime(frame[time_field])
        valid = stamps.dropna()
        if valid.empty:
            raise SourceUnreadable(
                f"'{time_field}' 을 시각으로 해석할 수 있는 행이 없다.", subject=time_field
            )

        # 1) 파일 순서 그대로의 차이 — 여기서 역순이 드러난다.
        as_written = valid.reset_index(drop=True)
        deltas_written = as_written.diff().dt.total_seconds().dropna()
        out_of_order = int((deltas_written < 0).sum())

        # 2) 정렬한 뒤의 차이 — 여기서 중복과 공백이 드러난다.
        ordered = as_written.sort_values().reset_index(drop=True)
        deltas = ordered.diff().dt.total_seconds().dropna()
        positive = deltas[deltas > 0]

        duplicates = int((deltas == 0).sum())
        median_interval = float(np.median(positive)) if not positive.empty else 0.0

        gap_threshold = median_interval * self._gap_multiplier if median_interval else 0.0
        if gap_threshold > 0:
            gaps = positive[positive > gap_threshold]
            gap_count = int(len(gaps))
            longest_gap = float(gaps.max()) if gap_count else 0.0
        else:  # pragma: no cover - 표본이 1개뿐인 극단 케이스
            gap_count, longest_gap = 0, 0.0

        return TimeAxisMeasurement(
            field_name=time_field,
            record_count=int(len(valid)),
            first=pd.Timestamp(ordered.iloc[0]).to_pydatetime(),
            last=pd.Timestamp(ordered.iloc[-1]).to_pydatetime(),
            median_interval_seconds=median_interval,
            out_of_order_count=out_of_order,
            duplicate_timestamp_count=duplicates,
            gap_count=gap_count,
            longest_gap_seconds=longest_gap,
        )
