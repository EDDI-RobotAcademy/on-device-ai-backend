"""통계 기반 이상 탐지 어댑터. (실습 3-13)

학습이 없다. `fit` 도 `epoch` 도 없다.
train 분할에서 평균과 표준편차를 뽑고, 그 기준으로 나머지를 판정할 뿐이다.

**그 단순함이 이 어댑터의 요점이다.**
같은 창, 같은 라벨, 같은 혼동 행렬로 재서 신경망과 나란히 놓는다.
"""

from __future__ import annotations

import numpy as np

from domain.model.evaluation import ConfusionMatrix, EvaluationResult
from domain.model.statistical_baseline import DetectionMethod, DetectorSpec
from domain.model.training_data_ref import TrainingDataRef
from domain.model.windowing import WindowLabelPolicy
from infrastructure.ml.windowing import WindowedDataset, build_windows

ANOMALY = "이상"
NORMAL = "정상"


class StatisticalAnomalyDetector:
    """domain.model.ports.BaselineDetector 구현.

    기준은 **train 분할에서만** 뽑는다.
    전체로 뽑으면 시험지를 미리 본 것이 된다 (실습 3-8 과 같은 규율).
    """

    def evaluate(
        self,
        data: TrainingDataRef,
        spec: DetectorSpec,
        *,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
        normal_label: str,
        split: str = "test",
    ) -> EvaluationResult:
        dataset = build_windows(
            data,
            window_length=window_length,
            stride=stride,
            label_policy=label_policy,
        )
        flagged = self._flag(dataset, spec, split=split)

        actual = self._actual_anomaly(dataset, split, normal_label)
        matrix = ConfusionMatrix.from_pairs(
            (NORMAL, ANOMALY),
            [
                (ANOMALY if a else NORMAL, ANOMALY if p else NORMAL)
                for a, p in zip(actual, flagged, strict=True)
            ],
        )
        return EvaluationResult(split=f"{split}/{spec.method.value}", matrix=matrix)

    def collapse_model_predictions(
        self,
        labels: tuple[str, ...],
        actual: np.ndarray,
        predicted: np.ndarray,
        *,
        normal_label: str,
    ) -> ConfusionMatrix:
        """학습 모델의 다중 분류 결과를 '이상 여부'로 접는다.

        접지 않고 비교하면 공평하지 않다 —
        통계 기반은 애초에 유형을 답하지 않기 때문이다.
        """
        return ConfusionMatrix.from_pairs(
            (NORMAL, ANOMALY),
            [
                (
                    ANOMALY if labels[int(a)] != normal_label else NORMAL,
                    ANOMALY if labels[int(p)] != normal_label else NORMAL,
                )
                for a, p in zip(actual, predicted, strict=True)
            ],
        )

    # -- 내부 --------------------------------------------------------------
    def _flag(
        self, dataset: WindowedDataset, spec: DetectorSpec, *, split: str
    ) -> list[bool]:
        train = dataset.features["train"]
        target = dataset.features[split]
        if train.size == 0 or target.size == 0:
            return [False] * int(target.shape[0])

        if spec.method is DetectionMethod.EWMA:
            per_sample = self._ewma_flags(train, target, spec)
        elif spec.method is DetectionMethod.IQR:
            per_sample = self._iqr_flags(train, target, spec)
        else:
            per_sample = self._sigma_flags(train, target, spec)

        # 창 안에서 걸린 비율이 기준을 넘으면 그 창을 '이상'으로 본다.
        ratio = per_sample.mean(axis=1)
        return [bool(r >= spec.min_flagged_ratio) for r in ratio]

    def _sigma_flags(
        self, train: np.ndarray, target: np.ndarray, spec: DetectorSpec
    ) -> np.ndarray:
        flat = train.reshape(-1, train.shape[-1])
        center = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        deviation = np.abs(target - center) / scale
        return (deviation > spec.threshold).any(axis=-1)

    def _iqr_flags(
        self, train: np.ndarray, target: np.ndarray, spec: DetectorSpec
    ) -> np.ndarray:
        flat = train.reshape(-1, train.shape[-1])
        q1 = np.percentile(flat, 25, axis=0)
        q3 = np.percentile(flat, 75, axis=0)
        iqr = np.where((q3 - q1) > 0, q3 - q1, 1.0)
        low = q1 - spec.threshold * iqr
        high = q3 + spec.threshold * iqr
        return ((target < low) | (target > high)).any(axis=-1)

    def _ewma_flags(
        self, train: np.ndarray, target: np.ndarray, spec: DetectorSpec
    ) -> np.ndarray:
        """기준선이 천천히 따라 움직인다.

        창 안에서 순차적으로 갱신한다 — 창 사이로는 이어지지 않는다.
        현장 구현이라면 이어져야 하지만, 여기서는 창 단위 평가와 맞추기 위해 끊는다.
        """
        flat = train.reshape(-1, train.shape[-1])
        scale = flat.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        start = flat.mean(axis=0)

        flags = np.zeros(target.shape[:2], dtype=bool)
        for i in range(target.shape[0]):
            level = start.copy()
            for t in range(target.shape[1]):
                value = target[i, t]
                flags[i, t] = bool(
                    (np.abs(value - level) / scale > spec.threshold).any()
                )
                level = spec.smoothing * value + (1.0 - spec.smoothing) * level
        return flags

    def _actual_anomaly(
        self, dataset: WindowedDataset, split: str, normal_label: str
    ) -> list[bool]:
        labels = dataset.labels
        return [
            labels[int(index)] != normal_label for index in dataset.targets[split]
        ]
