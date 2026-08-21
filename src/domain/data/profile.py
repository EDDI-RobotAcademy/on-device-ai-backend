"""데이터를 처음 열어봤을 때 보이는 사실(Value Object).

실습 1-1 "데이터를 열어보는 순간 현실이 보인다".

Profile 은 *판단이 없는 사실*이다.
"결측이 12% 다"까지가 Profile 이고,
"12% 는 학습 불가다"는 Policy 의 판단이다. 둘을 섞지 않는다.

이 값은 Infrastructure(pandas) 가 채워 넣지만, 타입은 Domain 이 소유한다.
그래야 pandas 를 polars 로 바꿔도 Domain 이 무너지지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.data.errors import UnknownField
from domain.shared.errors import InvariantViolation


class FieldType(Enum):
    """데이터가 스스로 주장하는 타입."""

    TIMESTAMP = "TIMESTAMP"
    REAL = "REAL"
    INTEGER = "INTEGER"
    CATEGORY = "CATEGORY"
    BOOLEAN = "BOOLEAN"
    TEXT = "TEXT"
    IMAGE_REF = "IMAGE_REF"
    UNKNOWN = "UNKNOWN"

    @property
    def is_numeric(self) -> bool:
        return self in (FieldType.REAL, FieldType.INTEGER)


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """열 하나에 대해 관측된 사실."""

    name: str
    inferred_type: FieldType
    total_count: int
    missing_count: int
    distinct_count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    stddev: float | None = None
    sample_values: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.total_count < 0:
            raise InvariantViolation("total_count 는 음수일 수 없다.", subject=self.name)
        if not 0 <= self.missing_count <= self.total_count:
            raise InvariantViolation(
                f"missing_count({self.missing_count}) 가 total_count({self.total_count}) 범위를 벗어났다.",
                subject=self.name,
            )
        if self.distinct_count < 0:
            raise InvariantViolation("distinct_count 는 음수일 수 없다.", subject=self.name)

    @property
    def present_count(self) -> int:
        return self.total_count - self.missing_count

    @property
    def missing_ratio(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.missing_count / self.total_count

    @property
    def is_constant(self) -> bool:
        """값이 하나뿐인 열. 센서가 죽었거나, 애초에 쓸모없는 열이다."""
        return self.present_count > 0 and self.distinct_count <= 1

    @property
    def is_all_missing(self) -> bool:
        return self.total_count > 0 and self.missing_count == self.total_count


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Dataset 을 한 번 훑어서 얻은 전체 사실."""

    row_count: int
    columns: tuple[ColumnProfile, ...]
    byte_size: int = 0

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise InvariantViolation("row_count 는 음수일 수 없다.", subject="row_count")
        if not self.columns:
            raise InvariantViolation(
                "열이 하나도 없는 데이터는 프로파일링의 의미가 없다.", subject="columns"
            )
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise InvariantViolation("중복된 열 이름이 있다.", subject="columns")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise UnknownField(f"프로파일에 '{name}' 열이 없다.", subject=name)

    def has_column(self, name: str) -> bool:
        return name in self.column_names

    @property
    def columns_with_missing(self) -> tuple[ColumnProfile, ...]:
        return tuple(c for c in self.columns if c.missing_count > 0)

    @property
    def constant_columns(self) -> tuple[ColumnProfile, ...]:
        return tuple(c for c in self.columns if c.is_constant)

    @property
    def worst_missing_ratio(self) -> float:
        return max((c.missing_ratio for c in self.columns), default=0.0)
