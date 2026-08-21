"""현장 디바이스 흉내내기. (실습 5-3)

**예측은 진짜다.** 모듈 4 에서 고른 그 결과물을, 모듈 5 의 현장 신호에 실제로 돌린다.
그래서 분포 이동도, 확신도 하락도 꾸며낸 숫자가 아니다.

**지연시간은 합성한다.** 이건 정직하게 밝혀야 한다.
PC 에서 아무리 재도 디바이스의 발열 감쇠(thermal throttling)를 재현할 수 없다.
그래서 "현장이라면 이렇게 됐을 것"을 만들어 넣는다.

    기준        모듈 4 의 벤치마크 × 디바이스 배수 (PC 보다 느리다)
    O05         DEV-02 는 3일차부터 팬이 죽어 점점 느려진다
    지터        가끔 다른 작업과 겹친다

운영에서의 창 자르기는 학습 때와 다르다.

    학습 (실습 3-8)   stride = window_length — 겹치면 분할이 샌다
    운영              **stride = 1** — 표본이 하나 들어올 때마다 다시 판단한다

디바이스는 분할을 하지 않는다. 샐 것이 없다.
"""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from domain.operations.inference_log import InferenceRecord
from infrastructure.analysis.table_loader import load_frame, numeric_view

Predict = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """디바이스 한 대의 성능 성질.

    이 숫자들은 데이터시트가 아니라 **현장 관찰**에서 온다.
    """

    device_id: str
    latency_multiplier: float = 8.0
    """PC 벤치마크 대비 배수. 임베디드 CPU 는 원래 느리다."""

    jitter_sigma: float = 0.12
    degradation_from_day: int | None = None
    """이 날(0부터 센다)**부터** 느려지기 시작한다. O05 — 팬 고장."""

    degradation_per_day: float = 1.8


DEFAULT_PROFILES: tuple[DeviceProfile, ...] = (
    DeviceProfile(device_id="DEV-01", latency_multiplier=8.0),
    DeviceProfile(
        device_id="DEV-02",
        latency_multiplier=8.4,
        degradation_from_day=3,  # 4일차부터
        degradation_per_day=2.6,
    ),
    DeviceProfile(device_id="DEV-03", latency_multiplier=7.6),
)


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    """무엇을 어떻게 돌릴 것인가."""

    stream_uri: str
    feature_fields: tuple[str, ...]
    label_field: str
    time_field: str = "timestamp"
    device_field: str = "device_id"
    window_length: int = 30
    stride: int = 1
    class_labels: tuple[str, ...] = ()
    normalization: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    baseline_p95_ms: float = 0.03
    labeled_ratio: float = 0.12
    profiles: tuple[DeviceProfile, ...] = DEFAULT_PROFILES
    seed: int = 20260520


class DeviceFleetSimulator:
    """현장 디바이스 여러 대가 추론하고 로그를 올리는 것을 흉내낸다."""

    def __init__(self, spec: SimulationSpec) -> None:
        self._spec = spec
        self._frame = self._load()

    # -- 조회 --------------------------------------------------------------
    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(self._frame[self._spec.device_field].unique())

    def run(self, predict: Predict, deployment_version: int) -> list[InferenceRecord]:
        """전 디바이스, 전 구간을 한 번에 돌린다."""
        records: list[InferenceRecord] = []
        for device_id in self.devices:
            records.extend(self._run_device(device_id, predict, deployment_version))
        records.sort(key=lambda r: (r.occurred_at, r.device_id))
        return records

    # -- 내부 --------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        spec = self._spec
        frame = load_frame(spec.stream_uri, "CSV").frame
        missing = [f for f in spec.feature_fields if f not in frame.columns]
        if missing:
            from infrastructure.errors import SourceUnreadable

            raise SourceUnreadable(f"입력 열이 없다: {missing}", subject=spec.stream_uri)
        return frame.sort_values(
            [spec.device_field, spec.time_field], kind="stable"
        ).reset_index(drop=True)

    def _run_device(
        self, device_id: str, predict: Predict, deployment_version: int
    ) -> list[InferenceRecord]:
        spec = self._spec
        rows = self._frame[self._frame[spec.device_field] == device_id].reset_index(
            drop=True
        )
        if len(rows) < spec.window_length:
            return []

        matrix = self._matrix(rows)
        starts = np.arange(0, len(rows) - spec.window_length + 1, spec.stride)
        batch = np.stack(
            [matrix[s : s + spec.window_length] for s in starts]
        ).astype("float32")

        logits = predict(batch)
        indices = logits.argmax(axis=1)
        confidence = _softmax_max(logits)

        # 창의 **마지막 표본** 시각이 그 추론의 시각이다.
        # 디바이스는 마지막 표본이 들어온 순간 판단한다.
        end_positions = starts + spec.window_length - 1
        timestamps = rows[spec.time_field].to_numpy()[end_positions]
        truths = rows[spec.label_field].to_numpy()[end_positions]

        latencies = self._latency(timestamps, device_id, len(starts))
        labeled = self._labeled_mask(device_id, len(starts))

        labels = spec.class_labels
        records: list[InferenceRecord] = []
        for position in range(len(starts)):
            digest = _digest(device_id, str(timestamps[position]))
            records.append(
                InferenceRecord(
                    occurred_at=str(timestamps[position]),
                    device_id=device_id,
                    deployment_version=deployment_version,
                    predicted_label=labels[int(indices[position])],
                    confidence=float(confidence[position]),
                    latency_ms=float(latencies[position]),
                    input_digest=digest,
                    ground_truth=(
                        str(truths[position]) if labeled[position] else None
                    ),
                )
            )
        return records

    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        """실습 1-7 에서 train 분할로 뽑은 통계를 **그대로** 쓴다.

        여기서 다시 계산하면 디바이스가 학습 때와 다른 전처리를 하는 것이 된다.
        그 순간 변환 동등성(실습 4-2)을 아무리 확인해도 소용없다.
        """
        spec = self._spec
        columns = []
        for name in spec.feature_fields:
            values = numeric_view(rows[name]).to_numpy(dtype="float64")
            stats = spec.normalization.get(name)
            if stats:
                mean, std = stats
                values = (values - mean) / (std if std else 1.0)
            columns.append(values)
        return np.column_stack(columns)

    def _latency(
        self, timestamps: np.ndarray, device_id: str, count: int
    ) -> np.ndarray:
        spec = self._spec
        profile = next(
            (p for p in spec.profiles if p.device_id == device_id),
            DeviceProfile(device_id=device_id),
        )
        rng = np.random.default_rng(spec.seed + _stable_hash(device_id))

        base = spec.baseline_p95_ms * profile.latency_multiplier
        latency = base * (1.0 + rng.normal(0.0, profile.jitter_sigma, count))

        if profile.degradation_from_day is not None:
            days = _day_index(timestamps)
            elapsed = np.clip(days - profile.degradation_from_day + 1, 0, None)
            latency *= 1.0 + elapsed * (profile.degradation_per_day - 1.0)

        # 가끔 다른 작업과 겹친다. 이것이 p95 와 p50 을 벌린다.
        spikes = rng.random(count) < 0.02
        latency[spikes] *= rng.uniform(2.5, 5.0, int(spikes.sum()))
        return np.clip(latency, 1e-4, None)

    def _labeled_mask(self, device_id: str, count: int) -> np.ndarray:
        """O06 — 정답은 일부에만 붙는다.

        무작위로 고른다. 현장에서는 작업자가 눈에 띈 것부터 확인하므로
        실제로는 무작위가 아니지만, 그 편향까지 흉내내지는 않는다.
        """
        rng = np.random.default_rng(
            self._spec.seed + 7919 + _stable_hash(device_id)
        )
        return rng.random(count) < self._spec.labeled_ratio


def _stable_hash(text: str) -> int:
    """파이썬의 hash() 는 실행마다 달라진다 (PYTHONHASHSEED).

    시드에 그것을 섞으면 같은 데이터에서 다른 지연시간이 나온다.
    실습이 매번 다른 숫자를 내면 문서에 숫자를 쓸 수 없다.
    """
    return zlib.crc32(text.encode()) % 10_000


def _softmax_max(logits: np.ndarray) -> np.ndarray:
    """확신도. 가장 큰 로짓의 softmax 값이다.

    로짓 자체를 확신도로 쓰면 모델마다 크기가 달라 비교가 안 된다.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return (exponent / exponent.sum(axis=1, keepdims=True)).max(axis=1)


def _day_index(timestamps: np.ndarray) -> np.ndarray:
    days = pd.to_datetime(pd.Series(timestamps)).dt.normalize()
    return (days - days.min()).dt.days.to_numpy()


def _digest(device_id: str, moment: str) -> str:
    """입력 지문. 원본 신호를 전부 남길 수는 없다.

    같은 입력이 다시 들어왔는지, 그때 그 입력을 다시 꺼낼 수 있는지에 쓴다.
    """
    return hashlib.sha1(
        f"{device_id}|{moment}".encode(), usedforsecurity=False
    ).hexdigest()[:16]


def stream_path(directory: Path) -> Path:
    return Path(directory) / "plant_power_operations.csv"
