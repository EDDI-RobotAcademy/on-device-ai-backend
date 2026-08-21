"""DistributionComparer 구현 (numpy/pandas). — 실습 1-9

PSI(Population Stability Index)
    학습 데이터의 분위수로 구간을 나눈 뒤, 현실 표본이 그 구간에 어떻게 떨어지는지 본다.

        PSI = Σ (obs% - ref%) × ln(obs% / ref%)

    관행적 해석: 0.1 미만 안정 / 0.1~0.25 이동 중 / 0.25 이상 심각.
    이 해석(임계값)은 Domain 의 RepresentativenessPolicy 가 갖고 있다. 여기서는 값만 낸다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.data.profile import FieldType
from domain.data.representativeness import (
    FieldDistributionShift,
    RepresentativenessMeasurement,
)
from domain.data.schema import DataSchema, FieldRole
from domain.data.source import DataSourceDescriptor
from infrastructure.analysis.table_loader import load_table, numeric_view

_EPSILON = 1e-6


class NumpyDistributionComparer:
    """domain.data.ports.DistributionComparer 구현."""

    def __init__(self, bin_count: int = 10) -> None:
        if bin_count < 2:
            raise ValueError("bin_count 는 2 이상이어야 한다.")
        self._bin_count = bin_count

    def compare(
        self,
        reference: DataSourceDescriptor,
        observed: DataSourceDescriptor,
        schema: DataSchema,
    ) -> RepresentativenessMeasurement:
        reference_frame = load_table(reference).frame
        observed_frame = load_table(observed).frame

        shifts: list[FieldDistributionShift] = []
        for spec in schema.feature_fields:
            if not spec.type.is_numeric:
                continue
            if spec.name not in reference_frame.columns:
                continue
            if spec.name not in observed_frame.columns:
                continue

            ref_values = numeric_view(reference_frame[spec.name]).dropna().to_numpy()
            obs_values = numeric_view(observed_frame[spec.name]).dropna().to_numpy()
            if ref_values.size == 0 or obs_values.size == 0:
                continue

            shifts.append(
                FieldDistributionShift(
                    field_name=spec.name,
                    psi=population_stability_index(
                        ref_values, obs_values, self._bin_count
                    ),
                    reference_mean=float(ref_values.mean()),
                    observed_mean=float(obs_values.mean()),
                    coverage_ratio=_coverage_ratio(ref_values, obs_values),
                )
            )

        return RepresentativenessMeasurement(
            reference_label=reference.collected_from,
            observed_label=observed.collected_from,
            field_shifts=tuple(shifts),
            unseen_category_ratio=_unseen_category_ratio(
                reference_frame, observed_frame, schema
            ),
            observed_sample_count=int(len(observed_frame)),
        )


def population_stability_index(
    reference: np.ndarray, observed: np.ndarray, bin_count: int = 10
) -> float:
    """학습 분포 대비 현실 분포의 이동량."""
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        # 학습 데이터가 상수였다. 현실도 같은 값이면 0, 아니면 완전히 벗어난 것이다.
        return 0.0 if np.allclose(observed, reference[0]) else float("inf")

    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    obs_counts, _ = np.histogram(observed, bins=edges)

    ref_ratio = np.maximum(ref_counts / reference.size, _EPSILON)
    obs_ratio = np.maximum(obs_counts / observed.size, _EPSILON)
    return float(np.sum((obs_ratio - ref_ratio) * np.log(obs_ratio / ref_ratio)))


def _coverage_ratio(reference: np.ndarray, observed: np.ndarray) -> float:
    """현실 표본 중 학습 데이터가 본 적 있는 값 범위 안에 들어오는 비율."""
    low, high = float(reference.min()), float(reference.max())
    inside = np.count_nonzero((observed >= low) & (observed <= high))
    return float(inside / observed.size)


def _unseen_category_ratio(
    reference: pd.DataFrame, observed: pd.DataFrame, schema: DataSchema
) -> float:
    """학습 데이터에 한 번도 없던 범주가 현실에서 차지하는 비율.

    GROUP 역할(LOT/배치 ID)은 세지 않는다. 그 값은 원래 매번 새로 생긴다.
    "오늘 LOT 번호가 처음 보는 값"은 사건이 아니다.
    "처음 보는 제품 코드"나 "학습에 없던 설비 호기"가 사건이다.
    """
    category_fields = [
        spec.name
        for spec in schema.fields
        if spec.type in (FieldType.CATEGORY, FieldType.TEXT)
        and spec.role in (FieldRole.FEATURE, FieldRole.METADATA)
        and spec.name in reference.columns
        and spec.name in observed.columns
        and _is_low_cardinality(reference[spec.name])
    ]
    if not category_fields:
        return 0.0

    total = len(observed)
    if total == 0:
        return 0.0

    unseen_mask = pd.Series(False, index=observed.index)
    for name in category_fields:
        known = set(reference[name].astype("string").dropna().unique())
        values = observed[name].astype("string")
        unseen_mask |= values.notna() & ~values.isin(known)

    return float(int(unseen_mask.sum()) / total)


def _is_low_cardinality(series: pd.Series, *, max_distinct: int = 50) -> bool:
    """값의 가짓수가 적어 '범주'라고 부를 수 있는가.

    행마다 값이 다른 열은 범주가 아니라 식별자다. 비교 대상이 아니다.
    """
    values = series.dropna()
    if values.empty:
        return False
    distinct = int(values.nunique())
    return distinct <= max_distinct and distinct / len(values) < 0.5
