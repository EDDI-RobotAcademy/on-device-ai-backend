"""실습 2-8 — 데이터 품질을 감이 아니라 숫자로 측정하라.

    pytest -m lesson_2_8 -s

여섯 축의 점수를 하나로 모은다. 그리고 그 점수를 **표본 수 언어로** 환산한다.
"COMPLETENESS 77점"보다 "학습 가능 표본이 8,640에서 8,261로 줄었다"가 현장에서 강하다.
"""

from __future__ import annotations

import pytest

from domain.data_quality.dimensions import QualityScore, deduct
from domain.data_quality.gate import DEFAULT_WEIGHTS, QualityGatePolicy
from domain.shared.errors import InvariantViolation
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_8


def test_여섯_축을_하나의_숫자로_모은다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-8 · 데이터 품질을 감이 아니라 숫자로 측정하라")

    view = qs.full_assessment(
        container,
        quality_container,
        dataset_id="dirty",
        assessment_id="qa-dirty",
        path=quality.dirty,
    )
    report.block("종합 점수", view.render())

    assert view.overall_score < 80
    assert view.grade in ("C", "D", "F")
    assert len(view.dimensions) == 6
    report.note(
        "축별 점수가 있어야 '무엇을 고칠지'가 나온다. "
        "종합 점수 하나만 있으면 '나쁘다'까지만 말할 수 있다."
    )


def test_점수는_학습_가능_표본_수로_환산된다(
    container, quality_container, quality
) -> None:
    view = qs.full_assessment(
        container,
        quality_container,
        dataset_id="dirty",
        assessment_id="qa-dirty",
        path=quality.dirty,
    )
    impact = view.impact
    assert impact is not None

    report.block("학습 관점 환산", impact.render())

    assert impact.total_rows == 8640
    assert impact.usable_rows < impact.total_rows
    assert impact.accuracy_ceiling < 1.0
    report.note(
        f"{impact.total_rows:,}행을 받았는데 실제로 쓸 수 있는 것은 "
        f"{impact.usable_rows:,}행이다."
    )
    report.note(
        f"그리고 라벨 오류 때문에 정확도 상한이 {impact.accuracy_ceiling:.1%} 다. "
        f"baseline 만 찍어도 {impact.baseline_accuracy:.1%} 가 나오므로, "
        "실제로 모델이 벌 수 있는 여지는 그 사이뿐이다."
    )


def test_감점_공식은_설명할_수_있어야_한다() -> None:
    """점수가 왜 그렇게 나왔는지 설명할 수 없으면 아무도 고치려 하지 않는다."""
    # 기준 이내면 감점 없음
    assert deduct(0.01, tolerance=0.02, cap=0.30, weight=45.0) == 0.0
    # 상한을 넘으면 가중치 전부
    assert deduct(0.50, tolerance=0.02, cap=0.30, weight=45.0) == pytest.approx(45.0)
    # 그 사이는 선형
    assert deduct(0.16, tolerance=0.02, cap=0.30, weight=45.0) == pytest.approx(22.5)

    report.block(
        "감점 공식",
        "  measured <= tolerance  → 0점 감점\n"
        "  measured >= cap        → weight 전부 감점\n"
        "  그 사이                 → 선형 비례",
    )
    report.note("복잡한 공식은 아무도 못 고친다. 단순한 것이 의도다.")


def test_가중치는_라벨_품질에_가장_무겁다() -> None:
    from domain.data_quality.dimensions import QualityDimension

    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    heaviest = max(DEFAULT_WEIGHTS.items(), key=lambda item: item[1])[0]
    assert heaviest is QualityDimension.LABEL_QUALITY

    report.block(
        "기본 가중치",
        "\n".join(
            f"  {dimension.value:<16}{weight:>6.0%}"
            for dimension, weight in sorted(
                DEFAULT_WEIGHTS.items(), key=lambda item: -item[1]
            )
        ),
    )
    report.note(
        "다른 축은 고칠 수 있다. 정답이 틀린 데이터는 아무리 고쳐도 그 이상 좋아지지 않는다."
    )


def test_가중치의_합이_1이_아니면_거부한다() -> None:
    with pytest.raises(InvariantViolation, match="합이"):
        QualityGatePolicy(weights={d: 0.5 for d in DEFAULT_WEIGHTS})


def test_측정하지_않은_축은_0점이_아니다(container, quality_container, quality) -> None:
    """'측정하지 않았다'와 '0점이다'는 다르다. 뭉뚱그리면 원인이 사라진다."""
    from application.data_quality.measure_completeness import (
        MeasureCompletenessCommand,
    )

    qs.prepare_dataset(container, "clean", quality.clean)
    qs.start(quality_container, "qa-partial", "clean")
    quality_container.measure_completeness().execute(
        MeasureCompletenessCommand(assessment_id="qa-partial")
    )
    view = qs.score(quality_container, "qa-partial")

    assert len(view.dimensions) == 1
    assert view.overall_score == pytest.approx(100.0)
    report.note(
        "한 축만 측정했고 그 축이 100점이라 종합도 100점이다. "
        "나머지를 0점으로 치지 않는다 — 그건 점수가 아니라 게이트가 막을 일이다."
    )


def test_점수_범위는_객체가_지킨다() -> None:
    with pytest.raises(InvariantViolation, match="0~100"):
        QualityScore(101.0)
    assert QualityScore(95.0).grade.value == "A"
    assert QualityScore(45.0).grade.value == "F"
