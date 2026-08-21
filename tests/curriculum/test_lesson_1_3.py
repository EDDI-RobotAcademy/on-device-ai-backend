"""실습 1-3 — 데이터의 정체부터 밝혀라.

    pytest -m lesson_1_3 -s

추론기에게 스키마 초안을 받아 본다. 그리고 그것이 왜 틀렸는지 확인한다.
데이터는 "내가 무엇인지"를 말해 주지 않는다. 타입만 말해 줄 뿐이다.
"""

from __future__ import annotations

import pytest

from application.data.infer_data_schema import InferDataSchemaCommand
from domain.data.schema import FieldRole
from tests.support import report
from tests.support.scenario import declare_schema, power_schema, profile, register

pytestmark = pytest.mark.lesson_1_3


def roles(draft) -> dict[str, str]:  # noqa: ANN001
    return {name: role for name, _, role in draft.fields}


def test_추론기가_만든_초안은_그럴듯하지만_틀렸다(container, power) -> None:
    report.section("실습 1-3 · 데이터의 정체부터 밝혀라")

    register(container, "raw", power.raw)
    profile(container, "raw")
    draft = container.infer_data_schema().execute(
        InferDataSchemaCommand(dataset_id="raw")
    )

    report.block("추론된 스키마 초안", draft.render())

    inferred = roles(draft)
    decided = {f.name: f.role.value for f in power_schema().fields}

    # 맞힌 것: 이름과 타입만으로 알 수 있는 것들
    assert inferred["timestamp"] == FieldRole.TIME_INDEX.value
    assert inferred["condition"] == FieldRole.LABEL.value
    report.note("맞힌 것: timestamp → TIME_INDEX, condition → LABEL")

    # 틀린 것: 현장 지식이 필요한 것들
    disagreement = {
        name: (inferred[name], decided[name])
        for name in decided
        if name in inferred and inferred[name] != decided[name]
    }
    assert disagreement, "추론이 전부 맞았다면 이 실습은 성립하지 않는다"

    report.block(
        "사람이 고쳐야 하는 지점",
        "\n".join(
            f"  {name:<20} 추론 {guessed:<12} → 확정 {final}"
            for name, (guessed, final) in sorted(disagreement.items())
        ),
    )

    # meter_id 는 이름이 'id' 로 끝난다는 이유로 식별자가 되었지만,
    # 실제로는 값이 하나뿐인 설비 메타데이터다.
    assert inferred["meter_id"] == FieldRole.IDENTIFIER.value
    assert decided["meter_id"] == FieldRole.METADATA.value

    # condition_review 는 2차 작업자의 판단이다. 입력도 정답도 아니다.
    assert "condition_review" in inferred


def test_추론기는_애매한_필드를_스스로_표시한다(container, power) -> None:
    register(container, "raw", power.raw)
    profile(container, "raw")
    draft = container.infer_data_schema().execute(
        InferDataSchemaCommand(dataset_id="raw")
    )
    assert draft.undecided_fields
    report.note(f"확인이 필요하다고 표시한 필드: {list(draft.undecided_fields)}")


def test_프로파일_없이는_추론할_수_없다(container, power) -> None:
    from application.shared.errors import UnsupportedOperation

    register(container, "raw", power.raw)
    with pytest.raises(UnsupportedOperation, match="먼저 열어봐야"):
        container.infer_data_schema().execute(InferDataSchemaCommand(dataset_id="raw"))


def test_확정된_스키마가_이후_모든_단계의_계약이_된다(container, power) -> None:
    """스키마를 확정하면 Dataset 이 그것을 계약으로 들고 간다."""
    from application.data.support import load_dataset

    register(container, "raw", power.raw)
    profile(container, "raw")
    declare_schema(container, "raw")

    dataset = load_dataset(container.repository, "raw")
    assert dataset.schema is not None
    assert dataset.schema.time_index.name == "timestamp"
    assert dataset.schema.label_field.name == "condition"
    assert [f.name for f in dataset.schema.group_fields] == ["batch_id"]
    assert len(dataset.schema.feature_fields) == 6
