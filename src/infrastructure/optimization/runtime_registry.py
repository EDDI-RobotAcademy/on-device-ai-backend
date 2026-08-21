"""변환된 결과물을 '실행할 수 있는 것'으로 들고 있는 곳.

각 런타임(PyTorch / TorchScript / ONNX Runtime / TFLite)은 API 가 전부 다르다.
그 차이를 여기서 **하나의 함수 모양**으로 흡수한다.

    predict(x: np.ndarray) -> np.ndarray

이 한 줄 덕분에 벤치마크·동등성 확인·정확도 평가가 런타임을 몰라도 된다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from domain.optimization.identifiers import ArtifactId


@dataclass(slots=True)
class LoadedRuntime:
    """실행 준비가 끝난 결과물."""

    predict: Callable[[np.ndarray], np.ndarray]
    input_shape: tuple[int, ...] = ()
    """표본 하나의 모양 (배치 축 제외). 벤치마크가 입력을 만들 때 쓴다."""

    activation_bytes: int = 0
    note: str = ""


@dataclass(slots=True)
class RuntimeRegistry:
    """artifact_id → 실행 가능한 것.

    실제 시스템이라면 디바이스에 올라간 모델이다.
    여기서는 메모리에 둔다 — 어디에 있든 Domain 은 모른다.
    """

    _items: dict[str, LoadedRuntime] = field(default_factory=dict)

    def put(self, artifact_id: ArtifactId, runtime: LoadedRuntime) -> None:
        self._items[str(artifact_id)] = runtime

    def get(self, artifact_id: ArtifactId) -> LoadedRuntime | None:
        return self._items.get(str(artifact_id))

    def require(self, artifact_id: ArtifactId) -> LoadedRuntime:
        runtime = self.get(artifact_id)
        if runtime is None:
            raise KeyError(f"실행 가능한 결과물이 없다: {artifact_id}")
        return runtime

    def clear(self) -> None:
        self._items.clear()
