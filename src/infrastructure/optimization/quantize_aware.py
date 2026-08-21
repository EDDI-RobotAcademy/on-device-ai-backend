"""PTQ 와 QAT 를 같은 자리에서 재는 어댑터. (실습 4-12)

TFLite 의 INT8 변환(실습 4-7)과 별개다.
저기서는 **변환 결과물**을 만들었고, 여기서는 **두 방법을 비교**한다.
비교를 위해서는 같은 격자로, 같은 평가 집합으로, 같은 코드가 눌러야 한다.

핵심은 세 줄이다.

    q = round(w / scale)          # 격자에 올린다
    q = clamp(q, -2^(b-1), ...)   # 표현 범위를 넘으면 자른다
    w' = q * scale                # 다시 실수로 돌린다

이걸 **순전파에만** 넣고 역전파에서는 통과시키는 것(straight-through)이
QAT 의 전부다. 그래야 기울기가 흐른다 — round 의 미분은 거의 모든 곳에서 0이다.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from domain.model.identifiers import ModelVersionId
from domain.model.tensor_spec import BatchSpec
from domain.model.training_config import TrainingConfig
from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.quantization import (
    QuantizationApproach,
    QuantizationComparison,
    QuantizationOutcome,
    QuantizationSpec,
)
from domain.shared.errors import InvariantViolation
from infrastructure.ml.torch_trainer import (
    TorchModelRegistry,
    TrainedModel,
    build_criterion,
    build_loaders,
    build_optimizer,
    run_epoch,
)


class _RoundStraightThrough(torch.autograd.Function):
    """반올림하되 기울기는 그대로 흘린다.

    round 의 미분은 거의 모든 점에서 0이다. 그대로 두면 아무것도 안 배운다.
    **역전파에서 항등함수인 척한다** — 이것이 straight-through estimator 다.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        return grad


def fake_quantize(weight: torch.Tensor, spec: QuantizationSpec) -> torch.Tensor:
    """가중치를 정수 격자에 올렸다가 다시 실수로 돌린다.

    값은 실수지만 **표현할 수 있는 값의 종류가 2^bits 로 줄어든다.**
    """
    limit = 2 ** (spec.bits - 1) - 1
    if spec.per_channel and weight.dim() > 1:
        flat = weight.reshape(weight.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp(min=1e-8) / limit
        shape = (-1,) + (1,) * (weight.dim() - 1)
        scale = scale.reshape(shape)
    else:
        scale = weight.abs().amax().clamp(min=1e-8) / limit

    quantized = _RoundStraightThrough.apply(weight / scale)
    return torch.clamp(quantized, -limit - 1, limit) * scale


class FakeQuantizedConv(nn.Module):
    """가중치를 눌러서 쓰는 층. 원래 층을 감싼다.

    **`inner.weight` 에 대입하지 않는다.** 대입하는 순간 계산 그래프가 끊기고,
    기울기가 원래 가중치까지 못 돌아간다 — 그러면 학습이 아무것도 안 한다.
    그래서 functional 형태로 부른다.
    """

    def __init__(self, inner: nn.Module, spec: QuantizationSpec) -> None:
        super().__init__()
        self.inner = inner
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = self.inner
        weight = fake_quantize(inner.weight, self.spec)

        if isinstance(inner, nn.Linear):
            return F.linear(x, weight, inner.bias)
        conv = F.conv1d if isinstance(inner, nn.Conv1d) else F.conv2d
        return conv(
            x,
            weight,
            inner.bias,
            inner.stride,
            inner.padding,
            inner.dilation,
            inner.groups,
        )


class TorchQuantizationLab:
    """domain.optimization.ports.QuantizationLab 구현. (실습 4-12)"""

    def __init__(self, models: TorchModelRegistry) -> None:
        self._models = models

    def compare(
        self,
        baseline: BaselineModelRef,
        *,
        bits: int,
        split: str = "test",
        epochs: int = 12,
        per_channel: bool = True,
    ) -> QuantizationComparison:
        trained = self._models.get(ModelVersionId.of(baseline.model_version_id))
        if trained is None:
            raise InvariantViolation(
                f"학습된 모델이 없다: {baseline.model_version_id}",
                subject="model_version_id",
            )

        baseline_accuracy, baseline_recall = _score(
            trained.module, trained, split, load_best=True
        )

        ptq_spec = QuantizationSpec(
            approach=QuantizationApproach.POST_TRAINING,
            bits=bits,
            per_channel=per_channel,
        )
        qat_spec = QuantizationSpec(
            approach=QuantizationApproach.QUANTIZATION_AWARE,
            bits=bits,
            per_channel=per_channel,
        )

        return QuantizationComparison(
            bits=bits,
            post_training=self._post_training(
                trained, ptq_spec, split, baseline_accuracy, baseline_recall
            ),
            quantization_aware=self._aware(
                trained, qat_spec, split, baseline_accuracy, baseline_recall, epochs
            ),
        )

    # -- 내부 --------------------------------------------------------------
    def _post_training(
        self,
        trained: TrainedModel,
        spec: QuantizationSpec,
        split: str,
        baseline_accuracy: float,
        baseline_recall: float,
    ) -> QuantizationOutcome:
        """다 배운 가중치를 그냥 누른다. **학습은 없다.**"""
        module = copy.deepcopy(trained.module)
        module.load_state_dict(trained.best_state)
        with torch.no_grad():
            for layer in _quantizable(module):
                layer.weight.copy_(fake_quantize(layer.weight, spec))

        accuracy, recall = _score(module, trained, split)
        return QuantizationOutcome(
            spec=spec,
            label="PTQ (변환 후 누름)",
            baseline_accuracy=baseline_accuracy,
            quantized_accuracy=accuracy,
            baseline_macro_recall=baseline_recall,
            quantized_macro_recall=recall,
            training_seconds=0.0,
            weight_bytes=_weight_bytes(module, spec),
        )

    def _aware(
        self,
        trained: TrainedModel,
        spec: QuantizationSpec,
        split: str,
        baseline_accuracy: float,
        baseline_recall: float,
        epochs: int,
    ) -> QuantizationOutcome:
        """누른 값으로 손실을 계산하며 배운다."""
        torch.manual_seed(42)
        module = copy.deepcopy(trained.module)
        module.load_state_dict(trained.best_state)
        _wrap_in_place(module, spec)

        config = TrainingConfig(
            epochs=epochs, batch_size=32, learning_rate=5e-4, seed=42
        )
        batch = BatchSpec(
            sample=trained.architecture.input_spec, batch_size=config.batch_size
        )
        loaders = build_loaders(trained.dataset, batch, config)
        criterion = build_criterion(trained.dataset, config)
        optimizer = build_optimizer(module, config)

        started = time.perf_counter()
        for _ in range(epochs):
            run_epoch(module, loaders["train"], criterion, optimizer)
        elapsed = time.perf_counter() - started

        # 학습이 끝나면 눌린 값을 **진짜로 굳힌다.** 배포되는 것은 이 가중치다.
        _unwrap_in_place(module)
        with torch.no_grad():
            for layer in _quantizable(module):
                layer.weight.copy_(fake_quantize(layer.weight, spec))

        accuracy, recall = _score(module, trained, split)
        return QuantizationOutcome(
            spec=spec,
            label="QAT (배우면서 누름)",
            baseline_accuracy=baseline_accuracy,
            quantized_accuracy=accuracy,
            baseline_macro_recall=baseline_recall,
            quantized_macro_recall=recall,
            training_seconds=elapsed,
            weight_bytes=_weight_bytes(module, spec),
        )


# ---------------------------------------------------------------------------
def _quantizable(module: nn.Module) -> list[nn.Module]:
    return [
        layer
        for layer in module.modules()
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear))
    ]


def _wrap_in_place(module: nn.Module, spec: QuantizationSpec) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            setattr(module, name, FakeQuantizedConv(child, spec))
        else:
            _wrap_in_place(child, spec)


def _unwrap_in_place(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, FakeQuantizedConv):
            setattr(module, name, child.inner)
        else:
            _unwrap_in_place(child)


def _weight_bytes(module: nn.Module, spec: QuantizationSpec) -> int:
    total = sum(layer.weight.numel() for layer in _quantizable(module))
    return int(total * spec.bits / 8)


def _score(
    module: nn.Module, trained: TrainedModel, split: str, *, load_best: bool = False
) -> tuple[float, float]:
    if load_best:
        module = copy.deepcopy(module)
        module.load_state_dict(trained.best_state)
    features = torch.from_numpy(trained.dataset.features[split])
    targets = trained.dataset.targets[split]
    module.eval()
    with torch.no_grad():
        predicted = module(features).argmax(dim=1).numpy()

    if not len(targets):
        return 0.0, 0.0
    accuracy = float((predicted == targets).mean())
    recalls = [
        float((predicted[targets == index] == index).mean())
        for index in np.unique(targets)
    ]
    return accuracy, float(np.mean(recalls))
