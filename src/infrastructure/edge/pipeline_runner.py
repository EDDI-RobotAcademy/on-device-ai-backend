"""디바이스 안에서 도는 다섯 단계를 실제로 돌린다. (실습 5-12, 5-13)

`device_simulator` 는 **로그를 만드는 것**이 목적이었다.
여기서는 **각 단계에서 무엇이 빠지는지 세는 것**이 목적이다.

무엇이 진짜이고 무엇이 합성인지:

    ACQUIRE      진짜 — CSV 의 행 수를 센다
    PREPROCESS   진짜 — 배치 경계를 넘는 창과 값이 빠진 창을 센다
    INFER        진짜 — 모듈 4 가 고른 결과물을 실제로 돌린다
    POSTPROCESS  진짜 — 실제 확신도로 보류를 판단한다
    EMIT         진짜 — AlertGate 규율을 실제로 적용한다

시간(duration_ms)만 실제 벽시계다. 디바이스 배수는 곱하지 않는다 —
여기서 보려는 것은 속도가 아니라 **어디서 빠지는가**이기 때문이다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from domain.operations.alerting import AlertGate, AlertLedger, AlertRule, Signal
from domain.operations.pipeline import (
    PipelineContract,
    PipelineRun,
    PipelineStage,
    StageOutcome,
)
from infrastructure.analysis.table_loader import load_frame, numeric_view

Predict = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """디바이스 한 대가 무엇을 어떻게 볼 것인가."""

    stream_uri: str
    device_id: str
    contract: PipelineContract
    label_field: str = "condition"
    stride: int = 3
    segment_field: str = "batch_id"
    """창이 이 값의 경계를 넘으면 **판단하지 않는다.**

    배치가 바뀌면 제품이 바뀐다. 한 창에 두 제품이 섞이면
    모델은 학습 때 본 적 없는 조합을 보게 된다 (실습 1-8, 3-4).
    **에러는 안 난다.** 그래서 여기서 세어 두지 않으면 아무도 모른다.
    """


class DevicePipelineRunner:
    """다섯 단계를 순서대로 돌리고 각 단계의 통과·탈락을 센다."""

    def __init__(self, spec: PipelineSpec) -> None:
        self._spec = spec

    def run(
        self, predict: Predict, rule: AlertRule
    ) -> tuple[PipelineRun, AlertLedger]:
        spec = self._spec
        contract = spec.contract

        # 1. ACQUIRE — 센서가 준 것
        started = time.perf_counter()
        frame: pd.DataFrame = load_frame(spec.stream_uri, "CSV").frame
        if "device_id" in frame.columns:
            frame = frame[frame["device_id"] == spec.device_id]
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        acquired = len(frame)
        acquire_ms = (time.perf_counter() - started) * 1000.0

        matrix = np.column_stack(
            [
                numeric_view(frame[name]).to_numpy(dtype="float64")
                for name in contract.feature_fields
            ]
        )
        unreadable = int(np.isnan(matrix).any(axis=1).sum())

        # 2. PREPROCESS — 창으로 자르고, 학습 때와 같은 통계로 정규화한다
        started = time.perf_counter()
        segments = _segments(frame, spec)
        normalized = _normalize(matrix, contract.normalization, contract.feature_fields)

        window = contract.input_shape[0]
        starts = [
            index
            for index in range(0, len(frame) - window + 1, spec.stride)
            # 창이 배치 경계를 넘으면 두 제품이 한 창에 섞인다 — 판단하지 않는다
            if segments[index] == segments[index + window - 1]
            and not np.isnan(normalized[index : index + window]).any()
        ]
        attempted_windows = max(0, (len(frame) - window) // spec.stride + 1)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        # 3. INFER — 모듈 4 가 고른 결과물을 실제로 돌린다
        started = time.perf_counter()
        if starts:
            batch = np.stack(
                [normalized[s : s + window] for s in starts]
            ).astype("float32")
            logits = predict(batch)
            probabilities = _softmax(logits)
            indices = probabilities.argmax(axis=1)
            confidences = probabilities.max(axis=1)
        else:
            indices = np.array([], dtype="int64")
            confidences = np.array([], dtype="float64")
        infer_ms = (time.perf_counter() - started) * 1000.0

        # 4. POSTPROCESS — 확신이 부족하면 답을 보류한다
        started = time.perf_counter()
        labels = contract.class_labels
        answered = confidences >= rule.min_confidence
        withheld = int((~answered).sum())
        postprocess_ms = (time.perf_counter() - started) * 1000.0

        # 5. EMIT — 알람 규율을 적용하고 로그를 남긴다
        started = time.perf_counter()
        stamps = pd.to_datetime(frame["timestamp"]).to_numpy()
        base = stamps[0].astype("datetime64[s]").astype("int64") if len(stamps) else 0
        signals = [
            Signal(
                at_seconds=float(
                    stamps[position + window - 1].astype("datetime64[s]").astype("int64")
                    - base
                ),
                label=labels[int(index)],
                confidence=float(confidence),
            )
            for position, index, confidence in zip(
                starts, indices, confidences, strict=True
            )
        ]
        ledger = AlertGate().apply(rule, signals)
        emit_ms = (time.perf_counter() - started) * 1000.0

        stages = (
            StageOutcome(
                stage=PipelineStage.ACQUIRE,
                attempted=acquired,
                succeeded=acquired - unreadable,
                duration_ms=acquire_ms,
                reason_counts={"판독 불가": unreadable} if unreadable else {},
            ),
            StageOutcome(
                stage=PipelineStage.PREPROCESS,
                attempted=attempted_windows,
                succeeded=len(starts),
                duration_ms=preprocess_ms,
                reason_counts={"배치 경계를 넘는 창": attempted_windows - len(starts)}
                if attempted_windows > len(starts)
                else {},
            ),
            StageOutcome(
                stage=PipelineStage.INFER,
                attempted=len(starts),
                succeeded=int(len(indices)),
                duration_ms=infer_ms,
            ),
            StageOutcome(
                stage=PipelineStage.POSTPROCESS,
                attempted=int(len(indices)),
                succeeded=int(answered.sum()),
                duration_ms=postprocess_ms,
                reason_counts={"확신 부족": withheld} if withheld else {},
            ),
            StageOutcome(
                stage=PipelineStage.EMIT,
                attempted=int(answered.sum()),
                succeeded=int(answered.sum()),
                duration_ms=emit_ms,
            ),
        )

        return (
            PipelineRun(
                device_id=spec.device_id,
                contract=contract,
                stages=stages,
                emitted_alerts=ledger.alert_count,
                withheld=withheld,
            ),
            ledger,
        )


# ---------------------------------------------------------------------------
def _segments(frame: pd.DataFrame, spec: PipelineSpec) -> np.ndarray:
    """각 행이 속한 구간(배치). 창이 이 경계를 넘으면 안 된다."""
    if spec.segment_field not in frame.columns:
        return np.zeros(len(frame), dtype="int64")
    return frame[spec.segment_field].astype("string").factorize()[0]


def _normalize(
    matrix: np.ndarray,
    statistics: Mapping[str, tuple[float, float]],
    fields: tuple[str, ...],
) -> np.ndarray:
    """**학습 때 쓰던 통계를 그대로 쓴다.** 다시 계산하면 그 순간 계약이 깨진다."""
    if not statistics:
        return matrix
    result = matrix.copy()
    for position, name in enumerate(fields):
        stats = statistics.get(name)
        if stats is None:
            continue
        center, scale = stats
        if scale > 0:
            result[:, position] = (result[:, position] - center) / scale
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def stream_of(directory: Path) -> Path:
    return directory / "plant_power_operations_stream.csv"
