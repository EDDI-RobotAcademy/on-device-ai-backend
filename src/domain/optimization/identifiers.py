"""Optimization Context 의 식별자."""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.identifier import Identifier


@dataclass(frozen=True, slots=True)
class OptimizationRunId(Identifier):
    """한 번의 최적화 시도."""


@dataclass(frozen=True, slots=True)
class ArtifactId(Identifier):
    """변환 결과물 하나. 런타임과 정밀도의 조합마다 하나씩 생긴다."""
