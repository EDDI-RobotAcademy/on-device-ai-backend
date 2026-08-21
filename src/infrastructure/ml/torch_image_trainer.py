"""이미지 학습 어댑터. (실습 3-11, 3-12)

`PyTorchModelTrainer` 와 학습 루프를 공유한다 (`torch_trainer` 의 부품들).
**학습은 같은 일이기 때문이다.** 다른 것은 재료를 만드는 방식뿐이다.

    시계열   CSV → 창으로 자르기 → (N, L, C)
    이미지   폴더 → 리사이즈·정규화 → (N, C, H, W)

그 아래로는 완전히 같은 코드가 돈다.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from domain.model.curve import EpochRecord
from domain.model.architecture import ModelArchitecture
from domain.model.identifiers import ModelVersionId
from domain.model.image_data_ref import ImageDataRef
from domain.model.protocol import SplitUsage
from domain.model.tensor_spec import BatchSpec
from domain.model.training_config import TrainingConfig
from infrastructure.ml.image_dataset import LabeledArrays, build_image_arrays
from infrastructure.ml.torch_architecture import build_module
from infrastructure.ml.torch_trainer import (
    TorchModelRegistry,
    TorchTrainingOutcome,
    TrainedModel,
    build_criterion,
    build_loaders,
    build_optimizer,
    run_epoch,
)


class PyTorchImageTrainer:
    """domain.model.ports.ImageModelTrainer 구현."""

    def __init__(self, registry: TorchModelRegistry | None = None) -> None:
        self.registry = registry or TorchModelRegistry()
        self._last_version: ModelVersionId | None = None
        self._last_arrays: LabeledArrays | None = None

    def train(
        self,
        data: ImageDataRef,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        batch: BatchSpec,
    ) -> TorchTrainingOutcome:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        arrays = build_image_arrays(data, seed=config.seed)
        self._last_arrays = arrays
        loaders = build_loaders(arrays, batch, config)

        module = build_module(architecture)
        criterion = build_criterion(arrays, config)
        optimizer = build_optimizer(module, config)

        records: list[EpochRecord] = []
        best_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        stale = 0

        for epoch in range(1, config.epochs + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = run_epoch(
                module, loaders["train"], criterion, optimizer
            )
            validation_loss, validation_accuracy = run_epoch(
                module, loaders["validation"], criterion, None
            )
            records.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    train_accuracy=train_accuracy,
                    validation_accuracy=validation_accuracy,
                    duration_seconds=time.perf_counter() - started,
                )
            )

            if validation_loss < best_loss - (
                config.early_stopping.min_delta if config.early_stopping else 0.0
            ):
                best_loss = validation_loss
                best_state = {
                    k: v.detach().clone() for k, v in module.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if config.early_stopping and stale >= config.early_stopping.patience:
                    break

        version_id = ModelVersionId.of(
            f"mv-{data.dataset_ref}-{architecture.kind.value.lower()}"
            f"-{architecture.pooling.value.lower()}-s{config.seed}"
        )
        self.registry.put(
            version_id,
            TrainedModel(
                module=module,
                dataset=arrays,
                architecture=architecture,
                best_state=best_state,
            ),
        )
        self._last_version = version_id

        return TorchTrainingOutcome(
            epochs=tuple(records),
            usage=SplitUsage(
                train_sample_count=int(arrays.features["train"].shape[0]),
                validation_sample_count=int(arrays.features["validation"].shape[0]),
                test_sample_count=int(arrays.features["test"].shape[0]),
                validation_evaluations=len(records),
                test_evaluations=0,
                # 이미지에는 창이 없다. 그래서 경계에서 새어 나가는 표본도 없다.
                overlapping_samples=0,
            ),
            artifact_uri=f"memory://models/{version_id}",
            labels=arrays.labels,
        )

    @property
    def last_version_id(self) -> ModelVersionId | None:
        return self._last_version

    @property
    def last_arrays(self) -> LabeledArrays | None:
        return self._last_arrays
