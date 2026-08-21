"""데이터를 텐서로 바꾸는 어댑터. (실습 3-1, 3-3, 3-4)

돌려주는 것은 텐서가 아니라 **텐서에 대한 요약**이다.
Domain 은 모양과 개수와 통계만 본다 — 배열은 여기 안에만 있다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from domain.model.tensor_spec import DatasetTensorSummary, ImageTensorSpec, TensorLayout
from domain.model.training_data_ref import TrainingDataRef
from domain.model.windowing import WindowLabelPolicy, WindowingSummary
from infrastructure.errors import SourceUnreadable
from infrastructure.ml.windowing import build_windows

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class TorchTensorMaterializer:
    """domain.model.ports.TensorMaterializer 구현."""

    def materialize(
        self,
        data: TrainingDataRef,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
    ) -> tuple[WindowingSummary, dict[str, DatasetTensorSummary]]:
        dataset = build_windows(
            data,
            window_length=window_length,
            stride=stride,
            label_policy=label_policy,
        )
        return dataset.summary, dataset.summaries()


class PillowImageTensorMaterializer:
    """domain.model.ports.ImageTensorMaterializer 구현. (실습 3-3)

    이미지를 숫자로 바꾸는 세 단계가 그대로 코드가 된다.

        1. 크기 조정   — 모델은 고정된 크기만 받는다
        2. 배열 변환   — (H, W, C) uint8 → (C, H, W) float32
        3. 정규화      — 채널마다 평균을 빼고 표준편차로 나눈다

    학습과 배포에서 이 셋이 **똑같아야** 한다. 그래서 계약(ImageTensorSpec)으로 고정한다.
    """

    def materialize(
        self, root_uri: str, spec: ImageTensorSpec
    ) -> dict[str, DatasetTensorSummary]:
        root = Path(root_uri)
        if not root.is_dir():
            raise SourceUnreadable(
                f"이미지 디렉터리를 찾을 수 없다: {root_uri}", subject=root_uri
            )

        arrays: list[np.ndarray] = []
        labels: list[str] = []
        unreadable = 0

        for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES):
            try:
                with Image.open(path) as image:
                    image.load()
                    converted = image.convert("L" if spec.channels == 1 else "RGB")
                    resized = converted.resize(
                        (spec.width, spec.height), Image.Resampling.BILINEAR
                    )
                    array = np.asarray(resized, dtype="float32") / 255.0
            except (UnidentifiedImageError, OSError, ValueError):
                unreadable += 1  # 실습 1-4 에서 찾았던 깨진 파일들이다
                continue

            if spec.channels == 1:
                array = array[:, :, None]
            array = (array - np.array(spec.mean, dtype="float32")) / np.array(
                spec.std, dtype="float32"
            )
            if spec.layout is not TensorLayout.CHANNEL_LAST:
                array = np.transpose(array, (2, 0, 1))

            arrays.append(array)
            labels.append(path.parent.name)

        if not arrays:
            raise SourceUnreadable(
                f"읽을 수 있는 이미지가 없다: {root_uri}", subject=root_uri
            )

        stacked = np.stack(arrays)
        counts = Counter(labels)
        return {
            "all": DatasetTensorSummary(
                split="all",
                sample_count=int(stacked.shape[0]),
                sample_shape=tuple(int(d) for d in stacked.shape[1:]),
                class_counts=dict(counts),
                feature_min=float(stacked.min()),
                feature_max=float(stacked.max()),
                feature_mean=float(stacked.mean()),
                feature_std=float(stacked.std()),
                nan_count=int(np.isnan(stacked).sum()),
            ),
            "unreadable": DatasetTensorSummary(
                split="unreadable",
                sample_count=unreadable,
                sample_shape=tuple(int(d) for d in stacked.shape[1:]),
            ),
        }
