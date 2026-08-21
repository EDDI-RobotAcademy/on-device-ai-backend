"""MissingValueMeasurer 구현 (pandas). — 실습 2-2

세는 것 네 가지.
    1. 결측 비율
    2. 연속 결측 최대 길이        → 센서 단절 구간
    3. 결측 집중도                → 특정 그룹에 몰렸는가 (무작위 결측이 아닌 증거)
    4. 비정상적으로 반복되는 값   → 결측을 채워 넣은 흔적 (은폐 결측)

4번이 이 측정기의 핵심이다. `fillna(0)` 한 줄이면 결측률은 0%가 되고,
그 0 은 물리 범위 안에 있어서 어떤 범위 검사도 통과한다.
"""

from __future__ import annotations

from itertools import groupby

import numpy as np
import pandas as pd

from domain.data_quality.completeness import (
    FieldMissingness,
    MissingValueMeasurement,
)
from domain.data_quality.target import AssessmentTarget
from infrastructure.analysis.table_loader import load_frame, numeric_view


class PandasMissingValueMeasurer:
    """domain.data_quality.ports.MissingValueMeasurer 구현."""

    def __init__(self, min_repeat_share: float = 0.005) -> None:
        """min_repeat_share 는 '이 정도 반복되면 들여다볼 가치가 있다'는 관측 하한이다.

        판정 기준이 아니다. 기준은 CompletenessPolicy 가 갖는다.
        """
        self._min_repeat_share = min_repeat_share

    def measure(self, target: AssessmentTarget) -> MissingValueMeasurement:
        frame = load_frame(target.uri, target.source_format).frame

        group = (
            frame[target.group_field].astype("string")
            if target.group_field and target.group_field in frame.columns
            else None
        )

        fields: list[FieldMissingness] = []
        for name in target.feature_fields:
            if name not in frame.columns:
                continue
            column = frame[name]
            missing_mask = column.isna()

            repeated_value, repeated_count = self._repeated_value(column)
            fields.append(
                FieldMissingness(
                    field_name=name,
                    total_count=int(len(column)),
                    missing_count=int(missing_mask.sum()),
                    longest_missing_run=_longest_run(missing_mask.to_numpy()),
                    concentration_ratio=_concentration(missing_mask, group),
                    repeated_value=repeated_value,
                    repeated_value_count=repeated_count,
                    repeated_value_mean_run=_mean_run(column, repeated_value),
                )
            )

        return MissingValueMeasurement(fields=tuple(fields))

    def _repeated_value(self, column: pd.Series) -> tuple[float | None, int]:
        """연속형 열에서 비정상적으로 자주 등장하는 '정확한' 값.

        연속형 센서 값이 같은 실수로 수백 번 반복될 확률은 사실상 0이다.
        그런 값이 있다면 사람이 넣은 것이다.
        """
        values = numeric_view(column).dropna()
        if values.empty or values.nunique() < 3:
            return None, 0
        counts = values.value_counts()
        top_value = float(counts.index[0])
        top_count = int(counts.iloc[0])
        if top_count / len(values) < self._min_repeat_share:
            return None, 0
        return top_value, top_count


def _mean_run(column: pd.Series, value: float | None) -> float:
    """그 값이 연속으로 이어진 평균 길이.

    이 숫자 하나가 은폐 결측과 진짜 물리 상태를 가른다.

        fillna(0) 로 채운 값   → 원래 결측이 있던 자리에 흩어진다. 평균 1.0 근처.
        설비 정지 중의 rpm=0   → 정지가 이어지는 동안 뭉친다. 평균이 크다.

    측정기는 이 숫자를 낼 뿐, '흩어졌다/뭉쳤다'의 기준선은 Policy 가 갖는다.
    """
    if value is None:
        return 1.0
    mask = (numeric_view(column) == value).fillna(False).to_numpy()
    runs = [sum(1 for _ in group) for present, group in groupby(mask) if present]
    if not runs:
        return 1.0
    return float(sum(runs) / len(runs))


def _longest_run(mask: np.ndarray) -> int:
    """True 가 연속으로 이어진 최대 길이."""
    longest = 0
    for value, group in groupby(mask):
        if value:
            longest = max(longest, sum(1 for _ in group))
    return int(longest)


TOP_GROUP_SHARE = 0.1


def _concentration(missing_mask: pd.Series, group: pd.Series | None) -> float:
    """결측이 소수의 그룹(LOT/설비)에 몰린 정도.

    **상위 10% 그룹이 전체 결측의 몇 %를 차지하는가**로 잰다.

        고르게 흩어져 있으면   → 0.1 근처 (상위 10% 가 10% 만 차지)
        세 개 LOT 에 몰려 있으면 → 0.8 근처

    한 그룹의 최대 비중만 보면, 세 개 LOT 에 나눠 몰린 경우를 놓친다.
    현장에서 수집 문제는 보통 '한 곳'이 아니라 '몇 곳'에서 난다.
    """
    total_missing = int(missing_mask.sum())
    if total_missing == 0 or group is None:
        return 0.0
    per_group = missing_mask.groupby(group, observed=True).sum().sort_values(
        ascending=False
    )
    if per_group.empty:
        return 0.0
    top_count = max(1, int(np.ceil(len(per_group) * TOP_GROUP_SHARE)))
    return float(per_group.iloc[:top_count].sum() / total_missing)
