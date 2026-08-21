"""OutlierMeasurer 구현 (pandas/numpy). — 실습 2-3

세 가지 척도를 **함께** 낸다. 하나만 쓰면 반드시 놓친다.

    z-score   평균·표준편차 기반. 이상치가 많으면 스스로 오염되어 눈이 먼다(masking).
    MAD       중앙값 기반. 강건하다. 오염되어도 계속 본다.
    변화율    값 하나만 보면 정상 범위 안이라 단변량으로는 절대 안 잡히는 것.

그리고 이상치를 라벨별로 나눠 센다.
'정상' 구간에 몰린 이상치는 지울 대상이 아니라 조사할 대상이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.data_quality.target import AssessmentTarget
from domain.data_quality.validity import FieldOutliers, OutlierMeasurement
from infrastructure.analysis.table_loader import load_frame, numeric_view

MAD_SCALE = 0.6745
"""정규분포에서 MAD 를 표준편차와 같은 눈금으로 맞추는 상수."""


class PandasOutlierMeasurer:
    """domain.data_quality.ports.OutlierMeasurer 구현."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        mad_threshold: float = 3.5,
        rate_sigma: float = 8.0,
    ) -> None:
        self._z_threshold = z_threshold
        self._mad_threshold = mad_threshold
        self._rate_sigma = rate_sigma

    def measure(self, target: AssessmentTarget) -> OutlierMeasurement:
        frame = load_frame(target.uri, target.source_format).frame
        labels = (
            frame[target.label_field].astype("string")
            if target.label_field and target.label_field in frame.columns
            else None
        )

        fields: list[FieldOutliers] = []
        for name in target.feature_fields:
            if name not in frame.columns:
                continue
            values = numeric_view(frame[name])
            present = values.dropna()
            if present.empty:
                fields.append(FieldOutliers(field_name=name, total_count=0))
                continue

            z_mask = _zscore_mask(values, self._z_threshold)
            mad_mask = _mad_mask(values, self._mad_threshold)

            physical = target.range_of(name)
            if physical is None:
                out_of_range = 0
            else:
                low, high = physical
                out_of_range = int(((present < low) | (present > high)).sum())

            outliers_by_label: dict[str, int] = {}
            if labels is not None:
                grouped = labels[mad_mask.fillna(False)].value_counts()
                outliers_by_label = {
                    str(key): int(count) for key, count in grouped.items()
                }

            fields.append(
                FieldOutliers(
                    field_name=name,
                    total_count=int(len(present)),
                    z_outlier_count=int(z_mask.sum()),
                    mad_outlier_count=int(mad_mask.sum()),
                    out_of_physical_range_count=out_of_range,
                    rate_violation_count=_rate_violations(values, self._rate_sigma),
                    max_abs_z=_max_abs_z(values),
                    outliers_by_label=outliers_by_label,
                )
            )

        # 라벨별 이상치 수를 그대로 넘긴다.
        # '무엇이 정상 라벨인가'는 Domain(ValidityPolicy)이 판단한다.
        return OutlierMeasurement(fields=tuple(fields))


def _zscore_mask(values: pd.Series, threshold: float) -> pd.Series:
    present = values.dropna()
    stddev = float(present.std(ddof=0))
    if stddev <= 0:
        return pd.Series(False, index=values.index)
    z = (values - float(present.mean())) / stddev
    return (z.abs() > threshold).fillna(False)


def _mad_mask(values: pd.Series, threshold: float) -> pd.Series:
    present = values.dropna()
    median = float(present.median())
    mad = float((present - median).abs().median())
    if mad <= 0:
        # 값의 절반 이상이 같은 값이면 MAD 가 0이 된다. 이때는 표준편차로 물러선다.
        return _zscore_mask(values, threshold)
    modified_z = MAD_SCALE * (values - median) / mad
    return (modified_z.abs() > threshold).fillna(False)


def _rate_violations(values: pd.Series, sigma: float) -> int:
    """직전 표본 대비 변화가 통상 변화폭의 sigma 배를 넘는 표본 수."""
    diffs = values.diff().abs().dropna()
    if diffs.empty:
        return 0
    typical = float(diffs.median())
    if typical <= 0:
        typical = float(diffs[diffs > 0].median()) if (diffs > 0).any() else 0.0
    if typical <= 0:
        return 0
    return int((diffs > typical * sigma).sum())


def _max_abs_z(values: pd.Series) -> float:
    present = values.dropna()
    stddev = float(present.std(ddof=0))
    if stddev <= 0 or present.empty:
        return 0.0
    return float(((present - float(present.mean())) / stddev).abs().max())
