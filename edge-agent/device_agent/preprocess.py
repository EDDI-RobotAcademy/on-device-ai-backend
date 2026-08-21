"""링 버퍼와 정규화. 파이프라인의 두 번째 단계다. (실습 5-12 PREPROCESS)

디바이스는 파일을 통째로 못 읽는다. **표본이 하나씩 들어온다.**
그래서 창(window)을 링 버퍼로 유지하고, 다 차면 한 판을 만든다.

두 가지를 여기서 지킨다.

    정규화 통계는 **학습 때 쓰던 값 그대로** (실습 1-7, 5-1)
        여기서 다시 계산하는 순간 디바이스가 다른 전처리를 하는 것이 된다.

    창이 구간 경계를 넘으면 **판단하지 않는다** (실습 5-12)
        배치가 바뀌면 제품이 바뀐다. 두 제품이 한 창에 섞이면
        모델은 학습 때 본 적 없는 조합을 본다. 에러는 안 난다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from domain.operations.pipeline import PipelineContract
from device_agent.sources import Sample


@dataclass(slots=True)
class WindowBuilder:
    """표본을 모아 창 하나를 만든다."""

    contract: PipelineContract
    stride: int = 1
    """운영은 stride=1 이다 — 표본이 하나 들어올 때마다 다시 판단한다.

    학습(실습 3-8)은 겹치면 분할이 새지만, 디바이스는 분할을 하지 않는다.
    """

    _buffer: deque[Sample] = field(default_factory=deque, init=False)
    _since_last: int = field(default=0, init=False)
    _scale: np.ndarray | None = field(default=None, init=False)
    _center: np.ndarray | None = field(default=None, init=False)

    dropped_segment_boundary: int = field(default=0, init=False)
    dropped_not_full: int = field(default=0, init=False)
    attempted: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        length = self.contract.input_shape[0]
        self._buffer = deque(maxlen=length)
        center = []
        scale = []
        for name in self.contract.feature_fields:
            stats = self.contract.normalization.get(name)
            center.append(stats[0] if stats else 0.0)
            scale.append(stats[1] if stats and stats[1] > 0 else 1.0)
        self._center = np.array(center, dtype="float32")
        self._scale = np.array(scale, dtype="float32")

    @property
    def window_length(self) -> int:
        return self.contract.input_shape[0]

    def offer(self, sample: Sample) -> np.ndarray | None:
        """표본 하나를 넣는다. 판단할 창이 완성되면 (1, L, C) 배열을 준다."""
        if len(sample.values) != len(self.contract.feature_fields):
            raise ValueError(
                f"표본의 열 수({len(sample.values)})가 계약"
                f"({len(self.contract.feature_fields)})과 다르다"
            )
        self._buffer.append(sample)
        self._since_last += 1

        if len(self._buffer) < self.window_length:
            self.dropped_not_full += 1
            return None
        if self._since_last < self.stride:
            return None
        self._since_last = 0
        self.attempted += 1

        first, last = self._buffer[0], self._buffer[-1]
        if first.segment != last.segment:
            self.dropped_segment_boundary += 1
            return None

        raw = np.array(
            [s.values for s in self._buffer], dtype="float32"
        )  # (L, C)
        normalized = (raw - self._center) / self._scale
        if not np.isfinite(normalized).all():
            self.dropped_not_full += 1
            return None
        return normalized[np.newaxis, ...]  # (1, L, C)

    @property
    def latest(self) -> Sample | None:
        return self._buffer[-1] if self._buffer else None
