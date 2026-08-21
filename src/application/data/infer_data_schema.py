"""InferDataSchema — 스키마 초안을 만든다. (실습 1-3)

추론은 어디까지나 초안이다.
"이 열이 라벨이다", "이 열이 설비 그룹이다"는 데이터가 알려줄 수 없다. 현장이 알려준다.
그래서 이 Use Case 는 Dataset 을 바꾸지 않는다. 사람이 보고 고칠 재료만 돌려준다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.support import load_dataset
from application.shared.errors import UnsupportedOperation
from domain.data.ports import DatasetRepository, SchemaInferrer
from domain.data.schema import DataSchema


@dataclass(frozen=True, slots=True)
class InferDataSchemaCommand:
    dataset_id: str


@dataclass(frozen=True, slots=True)
class SchemaDraftView:
    dataset_id: str
    fields: tuple[tuple[str, str, str], ...]
    """(필드명, 타입, 역할) — 역할은 추론값이며 반드시 사람이 검토해야 한다."""

    undecided_fields: tuple[str, ...]
    """추론기가 역할을 확신하지 못한 필드. 여기가 대화가 시작되는 지점이다."""

    def render(self) -> str:
        lines = [f"스키마 초안 ({self.dataset_id})", f"{'field':<20}{'type':<12}role"]
        lines.append("-" * 44)
        for name, type_name, role in self.fields:
            lines.append(f"{name:<20}{type_name:<12}{role}")
        if self.undecided_fields:
            lines.append("")
            lines.append(f"확인 필요: {', '.join(self.undecided_fields)}")
        return "\n".join(lines)

    @classmethod
    def of(cls, dataset_id: str, schema: DataSchema) -> SchemaDraftView:
        return cls(
            dataset_id=dataset_id,
            fields=tuple(
                (f.name, f.type.value, f.role.value) for f in schema.fields
            ),
            undecided_fields=tuple(
                f.name for f in schema.fields if f.role.value == "METADATA"
            ),
        )


class InferDataSchema:
    def __init__(
        self, repository: DatasetRepository, inferrer: SchemaInferrer
    ) -> None:
        self._repository = repository
        self._inferrer = inferrer

    def execute(self, command: InferDataSchemaCommand) -> SchemaDraftView:
        dataset = load_dataset(self._repository, command.dataset_id)
        if dataset.profile is None:
            raise UnsupportedOperation(
                "프로파일이 없다. 데이터를 먼저 열어봐야 스키마를 추론할 수 있다.",
                subject=str(dataset.id),
            )
        draft = self._inferrer.infer(dataset.profile)
        return SchemaDraftView.of(str(dataset.id), draft)
