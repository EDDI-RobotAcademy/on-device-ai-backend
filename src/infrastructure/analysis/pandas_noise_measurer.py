"""NoiseMeasurer 구현 (pandas/numpy). — 실습 2-6

신호를 추세와 잔차로 나눈다.

    추세  = 이동 중앙값 (튀는 값에 끌려가지 않는다)
    잔차  = 원신호 - 추세
    SNR   = 10·log10( var(추세) / var(잔차) )

이동 '평균'이 아니라 이동 '중앙값'을 쓰는 이유는,
평균은 이상치 하나에 추세 자체가 끌려가서 잔차가 과소평가되기 때문이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.data_quality.noise import FieldNoise, NoiseMeasurement
from domain.data_quality.target import AssessmentTarget
from infrastructure.analysis.table_loader import load_frame, numeric_view


class PandasNoiseMeasurer:
    """domain.data_quality.ports.NoiseMeasurer 구현."""

    def __init__(self, window: int = 9) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError("window 는 3 이상의 홀수여야 한다.")
        self._window = window

    def measure(self, target: AssessmentTarget) -> NoiseMeasurement:
        frame = load_frame(target.uri, target.source_format).frame

        # 시간축이 있으면 시간 순서대로 봐야 한다. 순서가 틀리면 잡음이 과대평가된다.
        if target.time_field and target.time_field in frame.columns:
            frame = frame.sort_values(target.time_field, kind="stable")

        fields: list[FieldNoise] = []
        for name in target.feature_fields:
            if name not in frame.columns:
                continue
            values = numeric_view(frame[name]).reset_index(drop=True)
            present = values.dropna()
            if len(present) < self._window * 2:
                continue

            trend = values.rolling(self._window, center=True, min_periods=1).median()
            residual = (values - trend).dropna()
            trend_present = trend.dropna()

            signal_power = float(trend_present.var(ddof=0))
            # 잔차의 '분산'을 쓰면 설비 정지 같은 진짜 사건 몇 개가 잡음으로 계산된다.
            # 실습 2-3 에서 배운 것과 같은 이유로 여기서도 강건 척도를 쓴다.
            noise_power = _robust_variance(residual.to_numpy())

            diffs = values.diff().dropna()
            total_var = float(present.var(ddof=0))
            high_frequency = (
                float(diffs.var(ddof=0) / (2 * total_var)) if total_var > 0 else 0.0
            )

            fields.append(
                FieldNoise(
                    field_name=name,
                    signal_power=max(signal_power, 0.0),
                    noise_power=max(noise_power, 0.0),
                    high_frequency_ratio=float(min(max(high_frequency, 0.0), 1.0)),
                    reversal_ratio=_reversal_ratio(diffs.to_numpy()),
                )
            )

        return NoiseMeasurement(fields=tuple(fields))


MAD_TO_SIGMA = 1.4826
"""정규분포에서 MAD 를 표준편차 눈금으로 맞추는 상수."""


def _robust_variance(residual: np.ndarray) -> float:
    """MAD 기반 잔차 분산.

    설비 정지처럼 드물고 큰 사건은 잔차를 크게 만든다.
    그것은 잡음이 아니라 신호다. 중앙값 기반 척도는 거기에 끌려가지 않는다.
    """
    if residual.size == 0:
        return 0.0
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if mad <= 0:
        return float(np.var(residual))
    return float((MAD_TO_SIGMA * mad) ** 2)


def _reversal_ratio(diffs: np.ndarray) -> float:
    """연속 차분의 부호가 뒤집히는 비율.

    깨끗한 신호는 한동안 같은 방향으로 움직인다.
    매 표본마다 방향이 바뀌면(톱니) 그것은 신호가 아니라 잡음이다.
    """
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (signs.size - 1))
