"""수집 주기를 바꿔 가며 원본을 다시 뽑아 본다. (실습 1-11)

**재기만 한다.** 이 주기로 모아도 되는지는 SamplingDesignPolicy 가 정한다.

여기서 하는 일은 하나다.
원본을 N배 간격으로 솎아 낸 뒤, 사건 구간이 몇 개나 살아남는지 센다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.data.sampling_design import SamplingObservation, SamplingPlan
from infrastructure.analysis.table_loader import load_frame, numeric_view
from infrastructure.errors import SourceUnreadable


class PandasSamplingProbe:
    """domain.data.ports.SamplingProbe 구현."""

    def probe(
        self,
        uri: str,
        source_format: str,
        *,
        time_field: str,
        label_field: str,
        normal_label: str,
        plan: SamplingPlan,
        value_field: str | None = None,
    ) -> SamplingObservation:
        frame = load_frame(uri, source_format).frame
        for name in (time_field, label_field):
            if name not in frame.columns:
                raise SourceUnreadable(f"'{name}' 열이 없다.", subject=name)

        frame = frame.sort_values(time_field, kind="stable").reset_index(drop=True)
        stamps = pd.to_datetime(frame[time_field], errors="coerce")
        base_interval = _base_interval(stamps)

        labels = frame[label_field].astype("string").str.strip()
        is_event = (labels != normal_label) & labels.notna()

        original_runs = _runs(is_event.to_numpy())
        run_lengths = sorted(len(run) for run in original_runs)
        shortest = run_lengths[0] * base_interval if run_lengths else 0.0
        typical = (
            run_lengths[len(run_lengths) // 2] * base_interval if run_lengths else 0.0
        )

        step = max(1, int(round(plan.interval_seconds / base_interval)))
        kept = np.zeros(len(frame), dtype=bool)
        kept[::step] = True

        sampled_event = is_event.to_numpy() & kept
        sampled_runs = _runs(sampled_event[kept])
        lost = sum(1 for run in original_runs if not sampled_event[run].any())

        distinct = 0
        if value_field and value_field in frame.columns:
            values = numeric_view(frame[value_field]).to_numpy(dtype="float64")[kept]
            values = values[np.isfinite(values)]
            if plan.value_resolution > 0:
                values = np.round(values / plan.value_resolution) * plan.value_resolution
            distinct = int(len(np.unique(values)))

        return SamplingObservation(
            interval_seconds=plan.interval_seconds,
            row_count=int(kept.sum()),
            event_row_count=int(sampled_event.sum()),
            event_run_count=len(sampled_runs),
            shortest_event_seconds=float(shortest),
            typical_event_seconds=float(typical),
            lost_event_runs=int(lost),
            distinct_value_count=distinct,
        )


def _base_interval(stamps: pd.Series) -> float:
    """원본의 표본 간격. 중앙값으로 잡는다 — 결측 하나에 흔들리지 않도록."""
    deltas = stamps.diff().dt.total_seconds().dropna()
    positive = deltas[deltas > 0]
    if positive.empty:
        raise SourceUnreadable("시간 간격을 잴 수 없다.", subject="time_field")
    return float(positive.median())


def _runs(flags: np.ndarray) -> list[np.ndarray]:
    """연속으로 True 인 구간들의 인덱스 배열."""
    if not flags.any():
        return []
    positions = np.flatnonzero(flags)
    breaks = np.flatnonzero(np.diff(positions) > 1) + 1
    return [run for run in np.split(positions, breaks) if run.size]
