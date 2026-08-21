"""PartitionEngine 구현 (pandas/numpy). — 실습 1-8

이 어댑터는 "나쁜 분할"도 그대로 수행한다.
막는 것은 Domain(PartitionPlan.validate_against)의 일이고,
여기서는 그 결과가 실제로 얼마나 새는지를 숫자로 보여 준다.
학생이 누수를 눈으로 봐야 규칙이 납득된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.data.partition import PartitionMeasurement, PartitionPlan, SplitStrategy
from domain.data.schema import DataSchema
from domain.data.source import DataSourceDescriptor
from infrastructure.analysis.table_loader import load_table, to_datetime
from infrastructure.errors import SourceUnreadable

_SPLITS = ("train", "validation", "test")


class PandasPartitionEngine:
    """domain.data.ports.PartitionEngine 구현."""

    def apply(
        self,
        source: DataSourceDescriptor,
        schema: DataSchema,
        plan: PartitionPlan,
        label_field: str | None = None,
    ) -> PartitionMeasurement:
        frame = load_table(source).frame
        if frame.empty:
            raise SourceUnreadable("분할할 행이 없다.", subject=source.uri)

        assignment = self._assign(frame, plan)

        group_field = plan.group_field or (
            schema.group_fields[0].name if schema.group_fields else None
        )
        overlapping = self._count_overlapping_groups(frame, assignment, group_field)
        time_overlap = self._time_overlap_seconds(frame, assignment, plan.time_field)

        distribution: dict[str, dict[str, int]] = {}
        if label_field and label_field in frame.columns:
            for split in _SPLITS:
                values = frame.loc[assignment == split, label_field]
                counts = values.astype("string").str.strip().value_counts()
                distribution[split] = {
                    str(name): int(count) for name, count in counts.items()
                }

        return PartitionMeasurement(
            train_count=int((assignment == "train").sum()),
            validation_count=int((assignment == "validation").sum()),
            test_count=int((assignment == "test").sum()),
            overlapping_group_count=overlapping,
            time_overlap_seconds=time_overlap,
            class_distribution=distribution,
        )

    # -- 분할 전략 ---------------------------------------------------------
    def _assign(self, frame: pd.DataFrame, plan: PartitionPlan) -> pd.Series:
        return assign_splits(frame, plan)

    # -- 누수 측정 ---------------------------------------------------------
    def _count_overlapping_groups(
        self, frame: pd.DataFrame, assignment: pd.Series, group_field: str | None
    ) -> int:
        if not group_field or group_field not in frame.columns:
            return 0
        groups = frame[group_field].astype("string")
        train_groups = set(groups[assignment == "train"].dropna())
        test_groups = set(groups[assignment == "test"].dropna())
        return len(train_groups & test_groups)

    def _time_overlap_seconds(
        self, frame: pd.DataFrame, assignment: pd.Series, time_field: str | None
    ) -> float:
        if not time_field or time_field not in frame.columns:
            return 0.0
        stamps = to_datetime(frame[time_field])
        train_times = stamps[assignment == "train"].dropna()
        test_times = stamps[assignment == "test"].dropna()
        if train_times.empty or test_times.empty:
            return 0.0
        overlap = (train_times.max() - test_times.min()).total_seconds()
        return float(max(overlap, 0.0))


# ---------------------------------------------------------------------------
# 분할 배정 — 다른 어댑터(정규화 통계 산출 등)도 같은 배정을 써야 한다.
# 여기가 한 곳이어야 "train 에서만 통계를 뽑았다"가 실제로 보장된다.
# ---------------------------------------------------------------------------
def assign_splits(frame: pd.DataFrame, plan: PartitionPlan) -> pd.Series:
    """각 행에 'train' / 'validation' / 'test' 를 배정한다."""
    if plan.strategy is SplitStrategy.TIME_ORDERED:
        if plan.time_field not in frame.columns:
            raise SourceUnreadable(
                f"'{plan.time_field}' 열이 원본에 없다.", subject=plan.time_field
            )
        order = to_datetime(frame[plan.time_field]).argsort(kind="stable")
        return _sequential_assign(frame.index, order, plan)

    if plan.strategy is SplitStrategy.GROUP_HOLDOUT:
        if plan.group_field not in frame.columns:
            raise SourceUnreadable(
                f"'{plan.group_field}' 열이 원본에 없다.", subject=plan.group_field
            )
        return _group_assign(frame, plan)

    rng = np.random.default_rng(plan.seed)
    order = rng.permutation(len(frame))
    return _sequential_assign(frame.index, order, plan)


def _sequential_assign(
    index: pd.Index, order: np.ndarray, plan: PartitionPlan
) -> pd.Series:
    total = len(order)
    train_end = int(total * plan.ratio.train)
    validation_end = train_end + int(total * plan.ratio.validation)

    labels = np.empty(total, dtype=object)
    positions = np.asarray(order)
    labels[positions[:train_end]] = "train"
    labels[positions[train_end:validation_end]] = "validation"
    labels[positions[validation_end:]] = "test"
    return pd.Series(labels, index=index)


def _group_assign(frame: pd.DataFrame, plan: PartitionPlan) -> pd.Series:
    """그룹 단위로 통째로 배정한다. 한 LOT 이 두 분할에 걸치지 않는다."""
    groups = frame[plan.group_field].astype("string")
    unique = pd.Index(groups.dropna().unique()).sort_values()
    rng = np.random.default_rng(plan.seed)
    shuffled = list(rng.permutation(np.asarray(unique, dtype=object)))

    count = len(shuffled)
    train_end = max(int(count * plan.ratio.train), 1 if count >= 3 else 0)
    validation_end = min(
        train_end + max(int(count * plan.ratio.validation), 1 if count >= 3 else 0),
        count,
    )

    mapping: dict[str, str] = {}
    for position, name in enumerate(shuffled):
        if position < train_end:
            mapping[str(name)] = "train"
        elif position < validation_end:
            mapping[str(name)] = "validation"
        else:
            mapping[str(name)] = "test"

    return groups.map(mapping).fillna("test")
