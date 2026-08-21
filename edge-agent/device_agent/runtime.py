"""TFLite 인터프리터. 파이프라인의 세 번째 단계다. (실습 5-12 INFER)

보드에서는 `tflite_runtime` 을, 개발 기계에서는 `ai_edge_litert` 를 쓴다.
**둘 다 없으면 즉시 실패한다** — 여기서 numpy 로 흉내내면
"모델이 도는 것처럼 보이는데 답이 다른" 상태가 된다.

스레드는 1로 고정한다. 보드에 코어가 하나일 수 있고 (실습 4-1, 4-13),
멀티코어로 잰 지연시간은 현장에서 그대로 나오지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


class RuntimeUnavailable(RuntimeError):
    """추론 런타임이 없다. **흉내내지 않는다.**"""


def _load_interpreter(model_path: str, num_threads: int):  # noqa: ANN202
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeUnavailable(
                "tflite_runtime 도 ai_edge_litert 도 없다. "
                "보드에는 tflite_runtime 을 올린다 — **흉내내는 대체 구현은 두지 않는다.**"
            ) from exc
    return Interpreter(model_path=model_path, num_threads=num_threads)


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    latency_ms: float
    scores: tuple[float, ...]


class TfliteRuntime:
    """모델 하나를 올려 두고 창 하나씩 돌린다."""

    def __init__(
        self,
        model_path: str,
        class_labels: tuple[str, ...],
        *,
        num_threads: int = 1,
    ) -> None:
        self._labels = class_labels
        self._interpreter = _load_interpreter(model_path, num_threads)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        # get_input_details() 가 준 dict 는 **그 시점의 사본**이다.
        # resize 해도 안 바뀐다 — 그래서 현재 모양을 따로 들고 있는다.
        self._shape = tuple(int(v) for v in self._input["shape"])

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._shape

    def predict(self, window: np.ndarray) -> Prediction:
        batch = np.ascontiguousarray(window, dtype="float32")
        if self._shape != batch.shape:
            self._interpreter.resize_tensor_input(self._input["index"], batch.shape)
            self._interpreter.allocate_tensors()
            self._shape = batch.shape

        started = time.perf_counter()
        self._interpreter.set_tensor(self._input["index"], batch)
        self._interpreter.invoke()
        scores = np.array(self._interpreter.get_tensor(self._output["index"]))[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        probabilities = _softmax(scores)
        index = int(np.argmax(probabilities))
        return Prediction(
            label=self._labels[index],
            confidence=float(probabilities[index]),
            latency_ms=latency_ms,
            scores=tuple(float(v) for v in probabilities),
        )


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()
