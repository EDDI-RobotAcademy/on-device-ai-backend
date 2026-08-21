"""시계열을 창으로 자른다. (실습 3-4)

여기서 하는 일은 셋이다.
    1. 길이 L, 간격 stride 로 창을 만든다
    2. 창 안의 라벨 분포로부터 창의 라벨을 정한다 (규칙은 Domain 이 준다)
    3. 시간 순서대로 train / validation / test 로 나눈다

3번이 중요하다. 창이 겹치면(stride < L) **분할 경계에서 표본이 새어 나간다.**
그 사실을 숨기지 않고 숫자로 돌려준다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from domain.model.tensor_spec import DatasetTensorSummary
from domain.model.training_data_ref import TrainingDataRef
from domain.model.windowing import WindowLabelPolicy, WindowingSummary
from infrastructure.analysis.table_loader import load_frame, numeric_view
from infrastructure.errors import SourceUnreadable

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class WindowedDataset:
    """창으로 자른 결과. 배열은 여기 안에만 있다."""

    features: Mapping[str, np.ndarray]
    """split → (N, L, C) float32."""

    targets: Mapping[str, np.ndarray]
    """split → (N,) int64."""

    labels: tuple[str, ...]
    summary: WindowingSummary
    boundary_overlap_samples: int

    @property
    def class_count(self) -> int:
        return len(self.labels)

    def summaries(self) -> dict[str, DatasetTensorSummary]:
        result: dict[str, DatasetTensorSummary] = {}
        for split in SPLITS:
            x, y = self.features[split], self.targets[split]
            counts = {
                self.labels[i]: int((y == i).sum()) for i in range(len(self.labels))
            }
            result[split] = DatasetTensorSummary(
                split=split,
                sample_count=int(x.shape[0]),
                sample_shape=tuple(int(d) for d in x.shape[1:]),
                class_counts=counts,
                feature_min=float(x.min()) if x.size else None,
                feature_max=float(x.max()) if x.size else None,
                feature_mean=float(x.mean()) if x.size else None,
                feature_std=float(x.std()) if x.size else None,
                nan_count=int(np.isnan(x).sum()),
            )
        return result


def build_windows(
    data: TrainingDataRef,
    *,
    window_length: int,
    stride: int,
    label_policy: WindowLabelPolicy,
) -> WindowedDataset:
    frame = load_frame(data.uri, data.source_format).frame
    if data.time_field and data.time_field in frame.columns:
        frame = frame.sort_values(data.time_field, kind="stable").reset_index(drop=True)

    missing = [f for f in data.feature_fields if f not in frame.columns]
    if missing:
        raise SourceUnreadable(f"입력 열이 없다: {missing}", subject=data.uri)
    if data.label_field not in frame.columns:
        raise SourceUnreadable(
            f"'{data.label_field}' 열이 없다.", subject=data.label_field
        )

    matrix = np.column_stack(
        [
            numeric_view(frame[name]).to_numpy(dtype="float64")
            for name in data.feature_fields
        ]
    )
    matrix = _normalize(matrix, data)
    raw_labels = frame[data.label_field].astype("string").str.strip().to_numpy()

    labels = _label_order(label_policy, raw_labels)
    index = {label: i for i, label in enumerate(labels)}

    starts = list(range(0, len(frame) - window_length + 1, stride))
    if not starts:
        raise SourceUnreadable(
            f"창({window_length})이 데이터({len(frame)}행)보다 길다.", subject=data.uri
        )

    windows = np.stack([matrix[s : s + window_length] for s in starts]).astype(
        "float32"
    )
    window_labels = np.array(
        [
            index[
                label_policy.label_for(
                    _count_labels(raw_labels[s : s + window_length])
                )
            ]
            for s in starts
        ],
        dtype="int64",
    )

    train_end, validation_end = _split_points(len(starts), data.split_ratio)
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(starts)),
    }

    features = {
        split: windows[low:high] for split, (low, high) in bounds.items()
    }
    targets = {
        split: window_labels[low:high] for split, (low, high) in bounds.items()
    }

    label_counts = {
        labels[i]: int((window_labels == i).sum()) for i in range(len(labels))
    }
    summary = WindowingSummary(
        source_row_count=int(len(frame)),
        window_length=window_length,
        stride=stride,
        window_count=len(starts),
        label_counts=label_counts,
        shared_sample_count=max(0, window_length - stride) * max(0, len(starts) - 1),
    )

    return WindowedDataset(
        features=features,
        targets=targets,
        labels=labels,
        summary=summary,
        boundary_overlap_samples=_boundary_overlap(
            starts, window_length, (train_end, validation_end)
        ),
    )


# ---------------------------------------------------------------------------
def _normalize(matrix: np.ndarray, data: TrainingDataRef) -> np.ndarray:
    """실습 1-7 에서 train 분할로 뽑아 둔 통계를 그대로 쓴다.

    여기서 다시 계산하지 않는다. 다시 계산하는 순간 학습과 배포가 어긋난다.
    """
    if not data.normalization:
        return matrix
    result = matrix.copy()
    for position, name in enumerate(data.feature_fields):
        stats = data.normalization.get(name)
        if stats is None:
            continue
        center, scale = stats
        if scale <= 0:
            continue
        result[:, position] = (result[:, position] - center) / scale
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _label_order(policy: WindowLabelPolicy, observed: np.ndarray) -> tuple[str, ...]:
    """라벨 순서를 고정한다. 순서가 바뀌면 혼동 행렬을 비교할 수 없다."""
    ordered: list[str] = []
    for label in policy.labels:
        if label not in ordered:
            ordered.append(label)
    for label in sorted({str(v) for v in observed if v and v == v}):
        if label not in ordered:
            ordered.append(label)
    return tuple(ordered)


def _count_labels(window: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(window, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


def _split_points(count: int, ratio: tuple[float, float, float]) -> tuple[int, int]:
    train_end = int(count * ratio[0])
    validation_end = train_end + int(count * ratio[1])
    return train_end, min(validation_end, count)


def _boundary_overlap(
    starts: list[int], window_length: int, cuts: tuple[int, int]
) -> int:
    """분할 경계에서 두 쪽이 공유하는 원본 표본 수. (실습 3-8)"""
    overlap = 0
    for cut in cuts:
        if cut <= 0 or cut >= len(starts):
            continue
        last_end = starts[cut - 1] + window_length
        next_start = starts[cut]
        overlap += max(0, last_end - next_start)
    return overlap


def frame_row_count(data: TrainingDataRef) -> int:
    frame: pd.DataFrame = load_frame(data.uri, data.source_format).frame
    return int(len(frame))
