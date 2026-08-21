"""DuplicateMeasurer 구현 (pandas). — 실습 2-7

**중복 판정은 입력(feature) 열로만 한다.**

타임스탬프는 모델의 입력이 아니다(실습 1-7).
따라서 타임스탬프만 다르고 입력 값이 같은 두 행은 모델에게 같은 표본이다.
모듈 1의 시간축 검사가 이것을 절대 잡지 못하는 이유다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.data_quality.target import AssessmentTarget
from domain.data_quality.uniqueness import DuplicateMeasurement
from infrastructure.analysis.table_loader import load_frame, numeric_view
from infrastructure.errors import SourceUnreadable


class PandasDuplicateMeasurer:
    """domain.data_quality.ports.DuplicateMeasurer 구현."""

    def __init__(self, round_digits: int = 6, near_tolerance: float = 1e-4) -> None:
        self._round_digits = round_digits
        self._near_tolerance = near_tolerance
        """near_tolerance 는 '사실상 같다'의 상대 기준이다."""

    def measure(self, target: AssessmentTarget) -> DuplicateMeasurement:
        frame = load_frame(target.uri, target.source_format).frame
        present = [f for f in target.feature_fields if f in frame.columns]
        if not present:
            raise SourceUnreadable(
                "입력 열이 원본에 하나도 없다.", subject=target.uri
            )

        if target.time_field and target.time_field in frame.columns:
            frame = frame.sort_values(target.time_field, kind="stable").reset_index(
                drop=True
            )

        features = frame[present].round(self._round_digits)
        duplicated_mask = features.duplicated(keep="first")
        exact_duplicates = int(duplicated_mask.sum())
        group_count = int(
            len(features[features.duplicated(keep=False)].drop_duplicates())
        )

        conflicting = 0
        if target.label_field and target.label_field in frame.columns:
            labels = frame[target.label_field].astype("string").str.strip()
            keyed = features.assign(_label=labels.to_numpy())
            grouped = keyed.groupby(present, dropna=False, observed=True)["_label"]
            distinct = grouped.nunique()
            conflicting_keys = distinct[distinct > 1].index
            if len(conflicting_keys) > 0:
                conflicting = int(grouped.size().loc[conflicting_keys].sum())

        return DuplicateMeasurement(
            total_rows=int(len(frame)),
            exact_duplicate_count=exact_duplicates,
            duplicate_group_count=group_count,
            near_duplicate_count=self._near_duplicates(frame, present),
            conflicting_label_count=conflicting,
        )

    def _near_duplicates(self, frame: pd.DataFrame, fields: list[str]) -> int:
        """인접 행과 사실상 같은 값인 행 수 (센서 홀드 / 재전송)."""
        if len(frame) < 2:
            return 0
        matrix = np.column_stack(
            [numeric_view(frame[name]).to_numpy(dtype="float64") for name in fields]
        )
        previous, current = matrix[:-1], matrix[1:]
        scale = np.maximum(np.abs(previous), 1.0)
        close = np.abs(current - previous) <= self._near_tolerance * scale
        comparable = ~np.isnan(previous) & ~np.isnan(current)
        # 비교 가능한 모든 열이 사실상 같으면 근접 중복으로 본다.
        rows_close = np.all(close | ~comparable, axis=1) & np.any(comparable, axis=1)
        return int(np.count_nonzero(rows_close))
