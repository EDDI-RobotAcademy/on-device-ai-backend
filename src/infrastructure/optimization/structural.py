"""구조 축소 어댑터 — 폭 줄이기, 가지치기. (실습 4-11)

여기서 실제로 세는 것은 넷이다.

    파라미터 수     0이 된 것도 여전히 파라미터다
    0 아닌 것       가지치기의 성과는 여기에 나타난다
    곱셈 횟수       **속도 이득의 상한**
    파일 크기       실제로 저장해 보고 잰다

넷을 따로 세는 이유는, 넷이 **따로 움직이기 때문**이다.
비구조적 가지치기에서는 두 번째만 줄고 나머지 셋은 그대로다.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from domain.model.architecture import ModelArchitecture
from domain.model.identifiers import ModelVersionId
from domain.model.training_config import TrainingConfig
from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.structural import (
    ReductionKind,
    StructuralOutcome,
    StructuralReduction,
)
from domain.shared.errors import InvariantViolation
from infrastructure.ml.torch_architecture import TorchArchitectureProfiler, build_module
from infrastructure.ml.torch_trainer import (
    TorchModelRegistry,
    TrainedModel,
    build_criterion,
    build_loaders,
    build_optimizer,
    run_epoch,
)


class TorchStructuralReducer:
    """domain.optimization.ports.StructuralReducer 구현."""

    def __init__(
        self, models: TorchModelRegistry, artifact_dir: Path | None = None
    ) -> None:
        self._models = models
        self._dir = artifact_dir or Path(".")
        self._profiler = TorchArchitectureProfiler()

    def reduce(
        self,
        baseline: BaselineModelRef,
        reduction: StructuralReduction,
        *,
        label: str,
        split: str = "test",
        fine_tune_epochs: int = 6,
    ) -> StructuralOutcome:
        trained = self._models.get(ModelVersionId.of(baseline.model_version_id))
        if trained is None:
            raise InvariantViolation(
                f"학습된 모델이 없다: {baseline.model_version_id}",
                subject="model_version_id",
            )

        before_profile = self._profiler.profile(trained.architecture)
        before_size = self._save_size(trained.module, f"{label}-before")
        before_accuracy = _accuracy(trained.module, trained, split)

        if reduction.kind.changes_shape:
            module, architecture = self._shrink(trained, reduction)
        else:
            module, architecture = self._prune(trained, reduction)

        if reduction.fine_tuned:
            _fine_tune(module, trained, fine_tune_epochs)

        after_profile = self._profiler.profile(architecture)
        after_size = self._save_size(module, f"{label}-after")
        after_accuracy, after_recall = _accuracy(
            module, trained, split, with_recall=True
        )

        total, nonzero = _weight_counts(module)
        return StructuralOutcome(
            reduction=reduction,
            label=label,
            parameter_count_before=before_profile.parameter_count,
            parameter_count_after=total,
            nonzero_parameter_count=nonzero,
            mac_count_before=before_profile.mac_count,
            mac_count_after=after_profile.mac_count,
            size_bytes_before=before_size,
            size_bytes_after=after_size,
            accuracy_before=before_accuracy,
            accuracy_after=after_accuracy,
            macro_recall_after=after_recall,
        )

    # -- 내부 --------------------------------------------------------------
    def _shrink(
        self, trained: TrainedModel, reduction: StructuralReduction
    ) -> tuple[nn.Module, ModelArchitecture]:
        """더 작은 구조를 **새로 만든다.**

        가중치를 물려받을 수 없다 — 모양이 다르기 때문이다.
        그래서 구조 축소는 재학습이 절차의 일부다.
        """
        # 새 구조는 무작위 초기값에서 시작한다.
        # 시드를 고정하지 않으면 "재학습 없이 얼마나 무너지는가"가 매번 달라진다.
        torch.manual_seed(42)

        old = trained.architecture
        if reduction.kind is ReductionKind.WIDTH:
            channels = tuple(
                max(1, int(round(c * (1.0 - reduction.ratio))))
                for c in old.hidden_channels
            )
        else:  # DEPTH
            keep = max(1, int(round(len(old.hidden_channels) * (1.0 - reduction.ratio))))
            channels = old.hidden_channels[:keep]
        architecture = old.with_channels(channels)
        return build_module(architecture), architecture

    def _prune(
        self, trained: TrainedModel, reduction: StructuralReduction
    ) -> tuple[nn.Module, ModelArchitecture]:
        """가중치를 0으로 만든다. **모양은 그대로다.**"""
        torch.manual_seed(42)
        module = copy.deepcopy(trained.module)
        module.load_state_dict(trained.best_state)

        targets = [
            layer
            for layer in module.modules()
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear))
        ]
        for layer in targets:
            if reduction.kind is ReductionKind.PRUNE_STRUCTURED and layer.weight.dim() > 1:
                prune.ln_structured(
                    layer, name="weight", amount=reduction.ratio, n=1, dim=0
                )
            else:
                prune.l1_unstructured(layer, name="weight", amount=reduction.ratio)
            prune.remove(layer, "weight")  # 마스크를 실제 가중치에 굳힌다
        return module, trained.architecture

    def _save_size(self, module: nn.Module, name: str) -> int:
        """실제로 저장해 보고 잰다. **0도 저장된다** — 그게 요점이다."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{name}.pt"
        torch.save(module.state_dict(), path)
        return path.stat().st_size


def _weight_counts(module: nn.Module) -> tuple[int, int]:
    total = nonzero = 0
    for parameter in module.parameters():
        total += parameter.numel()
        nonzero += int(torch.count_nonzero(parameter).item())
    return total, nonzero


def _fine_tune(module: nn.Module, trained: TrainedModel, epochs: int) -> None:
    """줄인 뒤에 다시 학습한다. 짧게. (실습 4-11)"""
    config = TrainingConfig(epochs=epochs, batch_size=32, learning_rate=1e-3, seed=42)
    torch.manual_seed(config.seed)
    from domain.model.tensor_spec import BatchSpec

    batch = BatchSpec(
        sample=trained.architecture.input_spec, batch_size=config.batch_size
    )
    loaders = build_loaders(trained.dataset, batch, config)
    criterion = build_criterion(trained.dataset, config)
    optimizer = build_optimizer(module, config)
    for _ in range(epochs):
        run_epoch(module, loaders["train"], criterion, optimizer)


def _accuracy(
    module: nn.Module, trained: TrainedModel, split: str, *, with_recall: bool = False
):  # noqa: ANN201
    features = torch.from_numpy(trained.dataset.features[split])
    targets = trained.dataset.targets[split]
    module.eval()
    with torch.no_grad():
        predicted = module(features).argmax(dim=1).numpy()

    accuracy = float((predicted == targets).mean()) if len(targets) else 0.0
    if not with_recall:
        return accuracy

    recalls = []
    for index in np.unique(targets):
        mask = targets == index
        recalls.append(float((predicted[mask] == index).mean()))
    return accuracy, float(np.mean(recalls)) if recalls else 0.0
