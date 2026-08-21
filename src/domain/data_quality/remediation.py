"""조치 — 데이터를 고쳤다는 기록.

데이터를 고치는 것은 **되돌릴 수 없는 행위**다.
누가, 왜, 무엇을, 몇 행이나 고쳤는지가 남지 않으면
석 달 뒤 아무도 그 데이터를 검증할 수 없다.

그래서 RemediationAction 은 근거 없이는 생성되지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.data_quality.dimensions import QualityDimension
from domain.shared.errors import InvariantViolation


class RemediationKind(Enum):
    DROP_ROWS = "DROP_ROWS"
    """못 믿을 구간을 잘라낸다. 가장 정직하고 가장 아깝다."""

    IMPUTE = "IMPUTE"
    """값을 채운다. 무작위 결측일 때만 정당하다."""

    CLIP = "CLIP"
    """범위 밖 값을 경계로 자른다. 정보가 사라진다는 것을 알고 써야 한다."""

    RELABEL = "RELABEL"
    """라벨을 고친다. 반드시 현장 확인을 거쳐야 한다."""

    DEDUPLICATE = "DEDUPLICATE"
    SMOOTH = "SMOOTH"
    """평활화. 과하면 이상 징후 자체가 사라진다."""

    RESAMPLE = "RESAMPLE"
    """재표집/가중치. 불균형 대응."""

    EXCLUDE_SEGMENT = "EXCLUDE_SEGMENT"
    """특정 구간/LOT 전체를 학습 대상에서 제외한다."""

    RECOLLECT = "RECOLLECT"
    """다시 수집한다. 가장 느리고 가장 확실하다."""


@dataclass(frozen=True, slots=True)
class RemediationAction:
    kind: RemediationKind
    dimension: QualityDimension
    target: str
    """무엇에 적용했는가. 필드명 또는 구간 설명."""

    affected_rows: int
    rationale: str
    """왜 이 조치를 선택했는가. 다른 선택지를 왜 택하지 않았는지까지."""

    decided_by: str

    def __post_init__(self) -> None:
        if self.affected_rows < 0:
            raise InvariantViolation(
                "영향받은 행 수는 음수일 수 없다.", subject="affected_rows"
            )
        if not self.target.strip():
            raise InvariantViolation("무엇에 적용했는지 없다.", subject="target")
        if len(self.rationale.strip()) < 5:
            raise InvariantViolation(
                f"'{self.kind.value}' 조치의 근거가 없다. "
                "근거 없는 데이터 수정은 조작과 구분되지 않는다.",
                subject="rationale",
            )
        if not self.decided_by.strip():
            raise InvariantViolation(
                "누가 결정했는지 없다. 나중에 되돌릴 근거가 없다.", subject="decided_by"
            )

    def describe(self) -> str:
        return (
            f"{self.kind.value} on {self.target} "
            f"({self.affected_rows:,}행) — {self.rationale} [{self.decided_by}]"
        )
