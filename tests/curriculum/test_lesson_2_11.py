"""실습 2-11 — 불균형을 줄이는 방법마다 잃는 것이 다르다.

    pytest -m lesson_2_11 -s

실습 2-5 에서 "불균형 데이터는 AI의 판단을 왜곡한다"를 **측정**했다.
그 다음에 오는 질문은 하나다.

    "그래서 어떻게 하죠?"

검색하면 "오버샘플링 하세요"가 나온다. 그런데 방법마다 대가가 다르다.

    복제      새 정보는 0이다. 모델은 그 몇 장을 외운다.
    버리기    비율은 맞지만 **정상의 다양성을 버린다.**
    가중치    데이터를 안 건드린다. 대신 **없는 것은 여전히 없다.**
    합성      없던 표본을 만든다. **현장에 없는 패턴도 만든다.**

그리고 어느 것을 쓰든 규칙이 하나 있다.
**분할한 뒤에, train 에만 적용한다.**
"""

from __future__ import annotations

import pytest

from application.data_quality.compare_rebalancing import CompareRebalancingCommand
from domain.data_quality.rebalancing import (
    RebalancingOutcome,
    RebalancingPlan,
    RebalancingPolicy,
    RebalancingStrategy,
)
from domain.shared.errors import InvariantViolation
from tests.support import quality_scenario as qs
from tests.support import report

pytestmark = pytest.mark.lesson_2_11

PLANS = (
    RebalancingPlan(RebalancingStrategy.NONE),
    RebalancingPlan(RebalancingStrategy.CLASS_WEIGHT),
    RebalancingPlan(RebalancingStrategy.OVERSAMPLE),
    RebalancingPlan(RebalancingStrategy.UNDERSAMPLE),
    RebalancingPlan(RebalancingStrategy.SYNTHETIC),
)


@pytest.fixture
def compared(container, quality_container, quality):  # noqa: ANN001, ANN201
    qs.prepare_dataset(container, "rebalance", quality.dirty)
    return quality_container.compare_rebalancing().execute(
        CompareRebalancingCommand(dataset_id="rebalance", plans=PLANS)
    )


def test_다섯_가지_방법을_나란히_놓는다(compared) -> None:
    report.section("실습 2-11 · 불균형을 줄이는 방법마다 잃는 것이 다르다")

    report.block("비교", compared.render())

    assert len(compared.outcomes) == 5
    none = compared.outcome_of("NONE")
    assert none.imbalance_before > 10
    report.note(
        "'아무것도 하지 않는다'도 하나의 전략이다. "
        "**그것과 비교하지 않으면 좋아졌는지 알 수 없다.**"
    )


def test_복제는_표본을_늘리지_않는다(compared) -> None:
    """이 실습의 본론."""
    over = compared.outcome_of("OVERSAMPLE")

    report.block(
        "오버샘플링이 한 일",
        f"  행 수         {over.total_before:,} → {over.total_after:,}\n"
        f"  불균형        {over.imbalance_before:.1f}배 → {over.imbalance_after:.1f}배\n"
        f"  복제한 행     {over.duplicated_rows:,}\n"
        f"  **서로 다른 표본**  {over.distinct_minority_samples:,} (그대로)\n"
        f"  실제로 늘어난 정보  {over.information_gain}",
    )

    assert over.imbalance_after < 1.5
    assert over.information_gain == 0
    report.note(
        "표에서는 비율이 완벽하게 맞았다. **그런데 새 정보는 한 줄도 늘지 않았다.** "
        "모델은 그 몇백 개를 통째로 외운다 — "
        "학습 정확도만 올라가고 검증은 안 따라온다 (실습 3-7)."
    )


def test_언더샘플링은_정상의_다양성을_버린다(compared) -> None:
    under = compared.outcome_of("UNDERSAMPLE")

    report.block(
        "언더샘플링이 한 일",
        f"  행 수      {under.total_before:,} → {under.total_after:,}\n"
        f"  버린 행    {under.discarded_rows:,} "
        f"({under.discarded_rows / under.total_before:.0%})",
    )

    assert under.discarded_rows > under.total_after
    assert under.verdict == "WARNED"
    report.note(
        "비율은 맞았다. 대신 데이터의 90% 이상이 사라졌다. "
        "**현장의 '정상'은 한 가지가 아니다** — "
        "버린 것 중에 다음 달에 필요할 정상이 있다. 그리고 되돌릴 수 없다."
    )


def test_가중치는_아무것도_잃지_않지만_아무것도_얻지도_않는다(compared) -> None:
    weight = compared.outcome_of("CLASS_WEIGHT")

    report.block(
        "가중치가 한 일",
        f"  행 수      {weight.total_before:,} → {weight.total_after:,} (그대로)\n"
        f"  복제 {weight.duplicated_rows} / 버림 {weight.discarded_rows} / "
        f"합성 {weight.synthesized_rows}\n"
        f"  불균형     {weight.imbalance_after:.1f}배 (그대로)",
    )

    assert weight.total_after == weight.total_before
    assert weight.verdict != "BLOCKED"
    report.note(
        "데이터를 안 건드렸으니 **시험지 유출도, 버린 것도 없다.** "
        "그래서 대개 여기서 시작한다 — 모듈 3 의 `class_weighted_loss` 가 이것이다. "
        "다만 40장으로 배운 것은 여전히 40장짜리다."
    )


def test_합성은_현장에_없는_값을_만든다(compared) -> None:
    synthetic = compared.outcome_of("SYNTHETIC")

    assert synthetic.synthesized_rows > 0
    assert "REBALANCE_SYNTHETIC" in [f.code for f in compared.findings]
    report.note(
        f"{synthetic.synthesized_rows:,}행을 만들어 냈다. "
        "**전류는 정상인데 전력만 이상한** 조합이 생길 수 있다. "
        "물리적으로 불가능한 표본을 배운 모델은 현장에서 이상한 곳에서 튄다 (실습 2-3)."
    )


def test_분할하기_전에_리샘플링하면_시험지가_유출된다(
    container, quality_container, quality
) -> None:
    """이 실습에서 가장 중요한 한 줄."""
    qs.prepare_dataset(container, "rebalance-leak", quality.dirty)
    view = quality_container.compare_rebalancing().execute(
        CompareRebalancingCommand(
            dataset_id="rebalance-leak",
            plans=(
                RebalancingPlan(RebalancingStrategy.OVERSAMPLE),
                RebalancingPlan(
                    RebalancingStrategy.OVERSAMPLE, applied_after_split=False
                ),
            ),
        )
    )

    after_split, before_split = view.outcomes
    report.block(
        "같은 전략, 적용 시점만 다르다",
        f"  분할 후 적용 : {after_split.verdict}\n"
        f"  분할 전 적용 : {before_split.verdict}",
    )

    assert after_split.verdict != "BLOCKED"
    assert before_split.verdict == "BLOCKED"
    assert "REBALANCE_BEFORE_SPLIT" in [f.code for f in view.findings]
    report.note(
        "복제된 표본이 train 과 test 양쪽에 들어간다. "
        "**정확도는 올라가고 현장에서는 그대로다.** "
        "지표가 좋아지기 때문에 아무도 의심하지 않는다 — "
        "이것이 이 실습에서 가장 중요한 한 줄이다."
    )


def test_어떤_리샘플링도_표본_부족을_고치지_못한다() -> None:
    outcome = RebalancingOutcome(
        strategy=RebalancingStrategy.OVERSAMPLE,
        before={"NORMAL": 8000, "FAULT": 12},
        after={"NORMAL": 8000, "FAULT": 8000},
        duplicated_rows=7988,
        distinct_minority_samples=12,
    )
    findings = RebalancingPolicy(min_distinct_minority=50).inspect(
        RebalancingPlan(RebalancingStrategy.OVERSAMPLE), outcome
    )

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(
        f.code == "REBALANCE_CANNOT_FIX_SHORTAGE" and f.is_blocking for f in findings
    )
    report.note(
        "FAULT 12장을 8,000장으로 늘렸다. 표에서는 완벽하다. "
        "**그러나 이 모델이 아는 고장은 여전히 12가지다.** "
        "비율을 맞춰서 지표가 좋아 보이면 "
        "'데이터를 더 모아야 한다'는 결론이 가려진다 — 그게 진짜 손해다."
    )


def test_목표_비율은_0과_1_사이여야_한다() -> None:
    with pytest.raises(InvariantViolation, match="0 초과 1 이하"):
        RebalancingPlan(RebalancingStrategy.OVERSAMPLE, target_ratio=1.5)
