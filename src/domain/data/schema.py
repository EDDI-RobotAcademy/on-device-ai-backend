"""데이터의 정체를 선언한다(Value Object).

실습 1-2 / 1-3 "CSV 하나에도 현장의 문제가 숨어 있다" / "데이터의 정체부터 밝혀라".

핵심 구분:
    inferred_type   — 파일이 그렇게 보인다는 관측 (DatasetProfile)
    DataSchema      — 우리가 그렇게 취급하겠다는 선언 (계약)

추론은 Infrastructure 가 도와줄 수 있지만, 확정은 사람이 한다.
확정된 Schema 는 이후 모든 단계(라벨/분할/학습/배포)의 계약이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.data.errors import UnknownField
from domain.data.inspection import Finding, InspectionKind, InspectionReport, Severity
from domain.data.profile import ColumnProfile, DatasetProfile, FieldType
from domain.shared.errors import InvariantViolation


class FieldRole(Enum):
    """이 열이 AI 파이프라인에서 맡는 역할."""

    TIME_INDEX = "TIME_INDEX"
    """시간축. 분할과 윈도우의 기준이 된다."""

    FEATURE = "FEATURE"
    """모델의 입력."""

    LABEL = "LABEL"
    """모델이 맞혀야 하는 정답."""

    IDENTIFIER = "IDENTIFIER"
    """개별 샘플 식별자. 학습 입력에 절대 넣으면 안 된다."""

    GROUP = "GROUP"
    """누수(leakage) 방지를 위해 분할 시 통째로 묶여야 하는 단위. 예: 설비 ID, 제품 LOT."""

    METADATA = "METADATA"
    """기록용. 학습에는 쓰지 않는다."""


@dataclass(frozen=True, slots=True)
class ValueRange:
    """물리적으로 가능한 값의 범위.

    전압이 -3000V 가 나올 수 없다는 지식은 데이터가 아니라 현장이 준다.
    """

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise InvariantViolation(
                f"minimum({self.minimum}) 이 maximum({self.maximum}) 보다 클 수 없다.",
                subject="ValueRange",
            )

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """열 하나에 대한 계약."""

    name: str
    type: FieldType
    role: FieldRole
    unit: str | None = None
    required: bool = True
    value_range: ValueRange | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvariantViolation("필드 이름은 비어 있을 수 없다.", subject="name")
        if self.value_range is not None and not self.type.is_numeric:
            raise InvariantViolation(
                f"'{self.name}' 은 수치형이 아니므로 물리 범위를 가질 수 없다.", subject=self.name
            )
        if self.role is FieldRole.TIME_INDEX and self.type is not FieldType.TIMESTAMP:
            raise InvariantViolation(
                f"'{self.name}' 은 TIME_INDEX 인데 타입이 TIMESTAMP 가 아니다.", subject=self.name
            )

    def accepts_profile(self, profile: ColumnProfile) -> bool:
        """관측된 타입이 선언과 호환되는가."""
        if profile.inferred_type is self.type:
            return True
        # INTEGER 로 관측된 열을 REAL 로 선언하는 것은 허용한다 (전류 12 → 12.0).
        if self.type is FieldType.REAL and profile.inferred_type is FieldType.INTEGER:
            return True
        # CATEGORY 와 TEXT 는 같은 문자열이고 **가짓수만 다르다.**
        # 추론기는 가짓수로 둘을 가르지만(50종 이하면 CATEGORY), 그건 휴리스틱이다.
        # LOT 이 72개인 열을 CATEGORY 라고 부르는 것은 틀린 선언이 아니다.
        if {self.type, profile.inferred_type} == {FieldType.CATEGORY, FieldType.TEXT}:
            return True
        # 전부 결측이면 타입을 관측할 수 없다. 타입 불일치로 몰아세우지 않는다.
        if profile.is_all_missing:
            return True
        return False


@dataclass(frozen=True, slots=True)
class DataSchema:
    """Dataset 의 구조 계약."""

    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.fields:
            raise InvariantViolation("빈 스키마는 계약이 아니다.", subject="fields")
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise InvariantViolation("중복된 필드 이름이 있다.", subject="fields")
        time_indexes = [f for f in self.fields if f.role is FieldRole.TIME_INDEX]
        if len(time_indexes) > 1:
            raise InvariantViolation(
                "시간축은 하나여야 한다. 두 개면 어느 쪽이 현장의 시간인지 알 수 없다.",
                subject="TIME_INDEX",
            )
        labels = [f for f in self.fields if f.role is FieldRole.LABEL]
        if len(labels) > 1:
            raise InvariantViolation(
                "이 Context 에서 라벨 필드는 하나로 제한한다.", subject="LABEL"
            )

    # -- 조회 -------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def field_of(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise UnknownField(f"스키마에 '{name}' 필드가 없다.", subject=name)

    def has_field(self, name: str) -> bool:
        return name in self.names

    def fields_with_role(self, role: FieldRole) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.role is role)

    @property
    def time_index(self) -> FieldSpec | None:
        found = self.fields_with_role(FieldRole.TIME_INDEX)
        return found[0] if found else None

    @property
    def label_field(self) -> FieldSpec | None:
        found = self.fields_with_role(FieldRole.LABEL)
        return found[0] if found else None

    @property
    def feature_fields(self) -> tuple[FieldSpec, ...]:
        return self.fields_with_role(FieldRole.FEATURE)

    @property
    def group_fields(self) -> tuple[FieldSpec, ...]:
        return self.fields_with_role(FieldRole.GROUP)

    # -- 판정 -------------------------------------------------------------
    def inspect(self, profile: DatasetProfile) -> InspectionReport:
        """선언(Schema)과 현실(Profile)을 맞대어 본다.

        실습 1-2 의 본체. 파일이 스키마를 배신하는 지점을 전부 드러낸다.
        """
        findings: list[Finding] = []

        for spec in self.fields:
            if not profile.has_column(spec.name):
                findings.append(
                    Finding(
                        code="FIELD_MISSING",
                        message="스키마가 요구한 필드가 파일에 없다.",
                        severity=Severity.CRITICAL if spec.required else Severity.WARNING,
                        subject=spec.name,
                    )
                )
                continue

            column = profile.column(spec.name)
            if not spec.accepts_profile(column):
                findings.append(
                    Finding(
                        code="TYPE_MISMATCH",
                        message=(
                            f"{spec.type.value} 로 선언했으나 "
                            f"{column.inferred_type.value} 로 관측된다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=spec.name,
                    )
                )
            if spec.required and column.is_all_missing:
                findings.append(
                    Finding(
                        code="REQUIRED_FIELD_EMPTY",
                        message="필수 필드인데 값이 전부 비어 있다.",
                        severity=Severity.CRITICAL,
                        subject=spec.name,
                        measured=1.0,
                        threshold=0.0,
                    )
                )
            if spec.value_range is not None and column.inferred_type.is_numeric:
                if column.minimum is not None and column.minimum < spec.value_range.minimum:
                    findings.append(
                        Finding(
                            code="BELOW_PHYSICAL_RANGE",
                            message="물리적으로 불가능한 최소값이 존재한다.",
                            severity=Severity.CRITICAL,
                            subject=spec.name,
                            measured=column.minimum,
                            threshold=spec.value_range.minimum,
                        )
                    )
                if column.maximum is not None and column.maximum > spec.value_range.maximum:
                    findings.append(
                        Finding(
                            code="ABOVE_PHYSICAL_RANGE",
                            message="물리적으로 불가능한 최대값이 존재한다.",
                            severity=Severity.CRITICAL,
                            subject=spec.name,
                            measured=column.maximum,
                            threshold=spec.value_range.maximum,
                        )
                    )

        declared = set(self.names)
        for column in profile.columns:
            if column.name not in declared:
                findings.append(
                    Finding(
                        code="UNDECLARED_FIELD",
                        message="파일에는 있으나 스키마가 모르는 열이다. 의미를 모르는 열은 학습에 넣지 않는다.",
                        severity=Severity.WARNING,
                        subject=column.name,
                    )
                )

        return InspectionReport(kind=InspectionKind.SCHEMA, findings=tuple(findings))
