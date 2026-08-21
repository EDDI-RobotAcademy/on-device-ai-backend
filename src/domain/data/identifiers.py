"""Data Context 의 식별자."""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.identifier import Identifier


@dataclass(frozen=True, slots=True)
class DatasetId(Identifier):
    """Dataset Aggregate 의 식별자."""


@dataclass(frozen=True, slots=True)
class PartitionId(Identifier):
    """Dataset 분할 결과의 식별자."""
