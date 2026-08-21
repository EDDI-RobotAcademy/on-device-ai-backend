"""실습 1-6 — 정상과 이상을 누가 결정하는가?

    pytest -m lesson_1_6 -s

라벨은 데이터에 들어 있지 않다. 사람이 넣는 것이다.
그래서 라벨 문제는 대부분 '정의가 없거나 사람마다 다른' 데서 나온다.
"""

from __future__ import annotations

import pytest

from domain.data.labeling import LabelDefinition, LabelSpace
from domain.shared.errors import InvariantViolation
from tests.support import report
from tests.support.scenario import (
    declare_schema,
    define_labels,
    profile,
    register,
)

pytestmark = pytest.mark.lesson_1_6


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def test_기준_없는_라벨은_만들어지지도_않는다() -> None:
    report.section("실습 1-6 · 정상과 이상을 누가 결정하는가?")

    with pytest.raises(InvariantViolation, match="판단 기준이 없다"):
        LabelDefinition(name="NG", meaning="", decided_by="품질팀")

    with pytest.raises(InvariantViolation, match="누가 정했는지"):
        LabelDefinition(
            name="NG", meaning="표면 균열 길이가 1mm 이상", decided_by=""
        )

    report.note(
        "'NG' 라는 이름만으로는 라벨이 아니다. "
        "다른 사람이 읽고 같은 판단을 내릴 수 있어야 라벨이다."
    )


def test_현장_라벨은_믿을_수_없는_상태다(container, power) -> None:
    register(container, "raw", power.raw)
    profile(container, "raw")
    declare_schema(container, "raw")
    view = define_labels(container, "raw")

    report.block("라벨 검증 (원본)", view.render())

    found = codes(view)
    assert "LABEL_UNDEFINED" in found      # 정의에 없는 'UNKNOWN'
    assert "LABEL_MISSING" in found        # 라벨이 빈 행
    assert "LABEL_DISAGREEMENT" in found   # 작업자끼리 판단이 다름
    assert "LABEL_IMBALANCED" in found     # 클래스 불균형
    assert view.verdict == "FAILED"

    disagreement = next(f for f in view.findings if f.code == "LABEL_DISAGREEMENT")
    assert disagreement.measured is not None
    report.note(
        f"작업자 일치율 {disagreement.measured:.1%} "
        f"(기준 {disagreement.threshold:.0%}) — "
        "모델이 배우는 것은 결함이 아니라 사람의 기준 차이다."
    )


def test_기준을_합의한_뒤에는_통과한다(container, power) -> None:
    register(container, "curated", power.curated)
    profile(container, "curated")
    declare_schema(container, "curated")
    view = define_labels(container, "curated")

    report.block("라벨 검증 (기준 합의 후)", view.render())

    found = codes(view)
    assert "LABEL_DISAGREEMENT" not in found
    assert "LABEL_UNDEFINED" not in found
    assert "LABEL_MISSING" not in found
    # 불균형은 남는다. 현장이 원래 그렇다. 다만 알고 가야 한다.
    assert found == {"LABEL_IMBALANCED"}
    assert view.verdict == "PASSED_WITH_WARNINGS"

    report.note("불균형은 데이터를 고쳐서 없앨 문제가 아니다. 알고 대응할 문제다.")


def test_교차_검토가_없으면_일치율을_주장할_수_없다(container, power) -> None:
    """검토를 한 적이 없는데 '일치율 100%' 라고 말하는 시스템이 가장 위험하다."""
    import pandas as pd

    from application.data.define_label_space import DefineLabelSpaceCommand
    from tests.support.scenario import condition_label_space, strict_label_policy

    frame = pd.read_csv(power.curated).drop(columns=["condition_review"])
    path = power.curated.parent / "no_review.csv"
    frame.to_csv(path, index=False)

    register(container, "no-review", path)
    profile(container, "no-review")

    from domain.data.schema import DataSchema

    from tests.support.scenario import power_schema

    schema = DataSchema(
        fields=tuple(
            f for f in power_schema().fields if f.name != "condition_review"
        )
    )
    declare_schema(container, "no-review", schema)

    view = container.define_label_space().execute(
        DefineLabelSpaceCommand(
            dataset_id="no-review",
            label_space=condition_label_space(),
            policy=strict_label_policy(),
        )
    )
    assert "LABEL_NO_CROSS_REVIEW" in codes(view)
    assert view.verdict == "FAILED"


def test_라벨_필드가_아닌_열을_라벨로_지정할_수_없다(container, power) -> None:
    from application.data.define_label_space import DefineLabelSpaceCommand
    from domain.data.errors import SchemaMismatch
    from tests.support.scenario import condition_label_space

    register(container, "raw", power.raw)
    profile(container, "raw")
    declare_schema(container, "raw")

    wrong = LabelSpace(
        field_name="batch_id", definitions=condition_label_space().definitions
    )
    with pytest.raises(SchemaMismatch, match="LABEL 이 아니다"):
        container.define_label_space().execute(
            DefineLabelSpaceCommand(dataset_id="raw", label_space=wrong)
        )
