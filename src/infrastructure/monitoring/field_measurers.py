"""현장 관측 어댑터. (실습 5-5, 5-6, 5-7, 5-9)

전부 **세기만 한다.** 좋은지 나쁜지는 한 줄도 말하지 않는다.
그 판단은 domain/operations 의 Policy 들이 한다.

    LogLatencyMeasurer      로그의 latency_ms 로 분위수를 낸다
    LogPredictionMixMeasurer 로그의 predicted_label 을 센다
    StreamInputDriftMeasurer 원본 신호를 학습 분포와 견준다  ← 유일하게 원본을 본다
    ReplayShadowRunner      같은 입력을 두 모델에 넣어 본다
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from domain.operations.drift import DriftReport, FeatureDrift
from domain.operations.identifiers import DeploymentId
from domain.operations.latency import LatencyProfile
from domain.operations.prediction_mix import PredictionMix
from domain.operations.shadow import ShadowRun
from domain.operations.window import ObservationWindow
from infrastructure.analysis.table_loader import load_frame, numeric_view
from infrastructure.monitoring.inference_log_store import InMemoryInferenceLogStore

PSI_BINS = 10


class LogLatencyMeasurer:
    """domain.operations.ports.LatencyMeasurer 구현. (실습 5-5)"""

    def __init__(
        self, logs: InMemoryInferenceLogStore, *, timeout_ms: float | None = None
    ) -> None:
        self._logs = logs
        self._timeout_ms = timeout_ms

    def measure(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> LatencyProfile:
        records = self._logs.records_in(deployment_id, window)
        values = np.array([r.latency_ms for r in records], dtype="float64")
        if values.size == 0:
            return LatencyProfile(
                window=window, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, max_ms=0.0
            )

        timeouts = 0
        if self._timeout_ms is not None:
            over = values > self._timeout_ms
            timeouts = int(over.sum())
            # 타임아웃은 분위수에서 뺀다. **끝나지 않은 추론에는 지연시간이 없다.**
            values = values[~over]
            if values.size == 0:
                values = np.array([self._timeout_ms])

        return LatencyProfile(
            window=window,
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            p99_ms=float(np.percentile(values, 99)),
            max_ms=float(values.max()),
            timeout_count=timeouts,
        )


class LogPredictionMixMeasurer:
    """domain.operations.ports.PredictionMixMeasurer 구현. (실습 5-6)"""

    def __init__(self, logs: InMemoryInferenceLogStore) -> None:
        self._logs = logs

    def measure(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> PredictionMix:
        records = self._logs.records_in(deployment_id, window)
        counts: dict[str, int] = {}
        confidence_sum: dict[str, float] = {}
        for record in records:
            label = record.predicted_label
            counts[label] = counts.get(label, 0) + 1
            confidence_sum[label] = confidence_sum.get(label, 0.0) + record.confidence

        return PredictionMix(
            window=window,
            counts=counts,
            mean_confidence={
                label: confidence_sum[label] / counts[label] for label in counts
            },
        )


class StreamInputDriftMeasurer:
    """domain.operations.ports.InputDriftMeasurer 구현. (실습 5-7)

    학습 때의 분포를 미리 받아 두고, 현장 구간을 거기에 견준다.
    **학습 분포를 남겨 두지 않으면 이 어댑터를 만들 수 없다.**
    그래서 배포 시점에 그것을 함께 내보내야 한다 (실습 5-1).
    """

    def __init__(
        self,
        stream_uri: str,
        *,
        feature_fields: Sequence[str],
        reference: Mapping[str, np.ndarray],
        time_field: str = "timestamp",
        device_field: str = "device_id",
    ) -> None:
        self._frame = load_frame(stream_uri, "CSV").frame
        self._fields = tuple(feature_fields)
        self._reference = dict(reference)
        self._time_field = time_field
        self._device_field = device_field

        self._edges: dict[str, np.ndarray] = {}
        self._reference_shares: dict[str, np.ndarray] = {}
        self._stats: dict[str, tuple[float, float]] = {}
        self._bounds: dict[str, tuple[float, float]] = {}
        for name, values in self._reference.items():
            self._prepare(name, np.asarray(values, dtype="float64"))

    def measure(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> DriftReport:
        rows = self._rows_in(window)
        if rows.empty:
            return DriftReport(window=window, features=())

        features = []
        for name in self._fields:
            if name not in self._reference or name not in rows.columns:
                continue
            current = numeric_view(rows[name]).to_numpy(dtype="float64")
            current = current[~np.isnan(current)]
            if current.size == 0:
                continue
            features.append(self._drift_of(name, current))

        return DriftReport(window=window, features=tuple(features))

    # -- 내부 --------------------------------------------------------------
    def _prepare(self, name: str, values: np.ndarray) -> None:
        values = values[~np.isnan(values)]
        if values.size == 0:
            return
        # 학습 분포를 10분위로 자른다. 실습 1-9 와 같은 방식이다.
        quantiles = np.linspace(0, 100, PSI_BINS + 1)
        edges = np.unique(np.percentile(values, quantiles))
        if edges.size < 2:
            edges = np.array([values.min(), values.min() + 1e-9])
        edges[0], edges[-1] = -np.inf, np.inf

        counts, _ = np.histogram(values, bins=edges)
        self._edges[name] = edges
        self._reference_shares[name] = _shares(counts)
        self._stats[name] = (float(values.mean()), float(values.std()) or 1.0)
        self._bounds[name] = (float(values.min()), float(values.max()))

    def _drift_of(self, name: str, current: np.ndarray) -> FeatureDrift:
        counts, _ = np.histogram(current, bins=self._edges[name])
        shares = _shares(counts)
        reference = self._reference_shares[name]

        psi = float(np.sum((shares - reference) * np.log(shares / reference)))

        mean, std = self._stats[name]
        low, high = self._bounds[name]
        return FeatureDrift(
            field_name=name,
            psi=max(psi, 0.0),
            mean_shift_sigma=float((current.mean() - mean) / std),
            out_of_range_ratio=float(((current < low) | (current > high)).mean()),
        )

    def _rows_in(self, window: ObservationWindow) -> pd.DataFrame:
        frame = self._frame
        moments = frame[self._time_field].astype("string")
        mask = (moments >= window.started_at) & (moments <= window.ended_at)
        if window.device_id is not None and self._device_field in frame.columns:
            mask &= frame[self._device_field] == window.device_id
        return frame[mask]


class ReplayShadowRunner:
    """domain.operations.ports.ShadowRunner 구현. (실습 5-9)

    현장에서는 디바이스가 두 모델을 동시에 돌린다.
    여기서는 같은 구간을 두 결과물로 다시 돌려 같은 결과를 얻는다.
    """

    def __init__(
        self,
        logs: InMemoryInferenceLogStore,
        *,
        replay,  # noqa: ANN001 - (window, artifact_id) → (labels, confidences, latency_ms)
        candidate_latency_ms: float,
        incumbent_label: str,
        candidate_label: str,
    ) -> None:
        self._logs = logs
        self._replay = replay
        self._candidate_latency_ms = candidate_latency_ms
        self._incumbent_label = incumbent_label
        self._candidate_label = candidate_label

    def run(
        self,
        deployment_id: DeploymentId,
        window: ObservationWindow,
        candidate_artifact_id: str,
    ) -> ShadowRun:
        records = self._logs.records_in(deployment_id, window)
        if not records:
            raise ValueError("이 구간에 로그가 없다. 비교할 입력이 없다.")

        candidate_labels = self._replay(window, candidate_artifact_id)
        incumbent_labels = [r.predicted_label for r in records]
        count = min(len(candidate_labels), len(incumbent_labels))
        candidate_labels = list(candidate_labels[:count])
        incumbent_labels = incumbent_labels[:count]
        selected = records[:count]

        agreement = sum(
            1 for a, b in zip(incumbent_labels, candidate_labels, strict=True) if a == b
        )
        latencies = np.array([r.latency_ms for r in selected], dtype="float64")

        labeled = [
            (index, record)
            for index, record in enumerate(selected)
            if record.ground_truth is not None
        ]
        incumbent_correct = sum(
            1 for index, record in labeled if incumbent_labels[index] == record.ground_truth
        )
        candidate_correct = sum(
            1 for index, record in labeled if candidate_labels[index] == record.ground_truth
        )

        return ShadowRun(
            window=window,
            incumbent_label=self._incumbent_label,
            candidate_label=self._candidate_label,
            sample_count=count,
            agreement_count=agreement,
            incumbent_p95_ms=float(np.percentile(latencies, 95)),
            candidate_p95_ms=self._candidate_latency_ms,
            incumbent_mix=_counts(incumbent_labels),
            candidate_mix=_counts(candidate_labels),
            labeled_count=len(labeled),
            incumbent_correct=incumbent_correct,
            candidate_correct=candidate_correct,
        )


def _shares(counts: np.ndarray) -> np.ndarray:
    """0 인 칸이 있으면 log 가 터진다. 아주 작은 값으로 받쳐 둔다."""
    total = counts.sum()
    shares = counts / total if total else np.full(counts.size, 1.0 / counts.size)
    return np.clip(shares, 1e-6, None)


def _counts(labels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
