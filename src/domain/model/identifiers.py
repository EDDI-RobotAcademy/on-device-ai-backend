"""Model Context 의 식별자."""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.identifier import Identifier


@dataclass(frozen=True, slots=True)
class TrainingRunId(Identifier):
    """한 번의 학습에 대한 식별자."""


@dataclass(frozen=True, slots=True)
class ModelVersionId(Identifier):
    """학습이 만들어 낸 산출물의 식별자.

    운영 단계(모듈 5)에서 "어느 모델이 배포되어 있는가"의 답이 되는 값이다.
    그래서 학습이 끝나기 전에는 존재하지 않는다.
    """
