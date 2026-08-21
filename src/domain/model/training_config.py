"""학습을 어떻게 돌릴 것인가. (실습 3-5)

여기 있는 값들은 취향이 아니다. 하나하나가 결과를 바꾸고,
바뀐 결과를 재현하려면 전부 기록되어 있어야 한다.

특히 `seed`. 이것이 없으면 "어제는 됐는데요"를 설명할 방법이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation


class Optimizer(Enum):
    SGD = "SGD"
    ADAM = "ADAM"


@dataclass(frozen=True, slots=True)
class EarlyStoppingRule:
    """언제 멈출 것인가. (실습 3-7 과 함께 읽는다)

    검증 손실이 좋아지지 않는데 계속 도는 것은 시간 낭비가 아니라,
    **외우는 시간을 벌어 주는 것**이다.
    """

    patience: int = 5
    min_delta: float = 1e-4
    monitor: str = "validation_loss"

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise InvariantViolation("patience 는 1 이상이어야 한다.", subject="patience")
        if self.min_delta < 0:
            raise InvariantViolation("min_delta 는 음수일 수 없다.", subject="min_delta")
        if self.monitor not in ("validation_loss", "validation_accuracy"):
            raise InvariantViolation(
                "monitor 는 validation_loss 또는 validation_accuracy 여야 한다.",
                subject="monitor",
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """한 번의 학습을 재현하는 데 필요한 전부."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: Optimizer = Optimizer.ADAM
    weight_decay: float = 0.0
    seed: int = 42
    class_weighted_loss: bool = False
    """불균형 데이터(실습 2-5)에서 소수 클래스를 무시하지 않게 하는 장치."""

    early_stopping: EarlyStoppingRule | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise InvariantViolation("epochs 는 1 이상이어야 한다.", subject="epochs")
        if self.batch_size < 1:
            raise InvariantViolation("batch_size 는 1 이상이어야 한다.", subject="batch_size")
        if not 0.0 < self.learning_rate < 1.0:
            raise InvariantViolation(
                f"learning_rate 는 0 과 1 사이여야 한다. (받은 값 {self.learning_rate})",
                subject="learning_rate",
            )
        if self.weight_decay < 0:
            raise InvariantViolation(
                "weight_decay 는 음수일 수 없다.", subject="weight_decay"
            )

    def describe(self) -> str:
        parts = [
            f"epochs={self.epochs}",
            f"batch={self.batch_size}",
            f"lr={self.learning_rate:g}",
            f"opt={self.optimizer.value}",
            f"seed={self.seed}",
        ]
        if self.class_weighted_loss:
            parts.append("class_weighted")
        if self.early_stopping:
            parts.append(f"early_stop(patience={self.early_stopping.patience})")
        return " ".join(parts)
