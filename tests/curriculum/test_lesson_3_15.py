"""실습 3-15 — 데이터를 고치면 모델이 얼마나 달라지는가.

    pytest -m lesson_3_15 -s

모듈 2 는 "이 데이터는 나쁘다"까지 말했다. 그런데 현장에서는 반드시 이 반문이 온다.

    "그래서 모델이 얼마나 나빠지는데요?"

품질 점수가 62점이라는 말로는 아무도 설득되지 않는다.
**같은 모델을 두 데이터로 학습시켜서 같은 시험지로 채점해 보면** 대답이 나온다.

여기서 조심할 것이 하나 있다.

    시험지는 두 학습 파일 어느 쪽과도 겹치지 않아야 한다.

정제된 파일의 일부로 채점하면 정제된 쪽이 자기 답안지를 본 셈이 된다.
그래서 **다른 날 데이터**를 따로 둔다.
"""

from __future__ import annotations

import pytest

from application.model.compare_experiments import (
    CompareExperimentsCommand,
    TrialRequest,
)
from tests.support import report

pytestmark = pytest.mark.lesson_3_15

TRIALS = (
    TrialRequest(
        "run-quality-dirty", "오염된 데이터", {"data": "dirty"}, split="holdout"
    ),
    TrialRequest(
        "run-quality-clean", "정제된 데이터", {"data": "clean"}, split="holdout"
    ),
)


def _board(quality_impact_lab):  # noqa: ANN001
    return quality_impact_lab.model.compare_experiments().execute(
        CompareExperimentsCommand(name="데이터 품질과 모델 성능", trials=TRIALS)
    )


def test_게이트가_먼저_답을_알고_있었다(quality_impact_lab) -> None:
    report.section("실습 3-15 · 데이터를 고치면 모델이 얼마나 달라지는가")

    dirty = quality_impact_lab.results["오염된 데이터"].gate
    clean = quality_impact_lab.results["정제된 데이터"].gate

    report.block(
        "Data Quality Gate (모듈 2)",
        f"  오염된 데이터 : {dirty.verdict}\n  정제된 데이터 : {clean.verdict}",
    )

    assert dirty.verdict == "FAILED"
    assert clean.verdict != "FAILED"
    report.note(
        "**학습을 돌리기 전에 이미 알 수 있었다.** "
        "이 실습은 그 게이트를 일부러 우회해서 '얼마나 나빠지는지'를 재는 자리다 — "
        "현장에서 이 우회를 하면 안 된다."
    )


def test_같은_시험지로_채점하면_차이가_보인다(quality_impact_lab) -> None:
    """이 실습의 본론."""
    view = _board(quality_impact_lab)
    report.block("비교표", view.render())

    dirty = view.trial_of("오염된 데이터")
    clean = view.trial_of("정제된 데이터")

    assert dirty.evaluated_samples == clean.evaluated_samples
    assert clean.macro_f1 > dirty.macro_f1 + 0.05
    report.note(
        f"같은 {clean.evaluated_samples}개로 채점했다. "
        f"macro F1 {dirty.macro_f1:.3f} → {clean.macro_f1:.3f}. "
        "**'품질 점수 62점'보다 이 한 줄이 설득력이 있다.**"
    )


def test_재현율은_비슷한데_망가진_것은_정밀도다(quality_impact_lab) -> None:
    view = _board(quality_impact_lab)
    dirty = view.trial_of("오염된 데이터")
    clean = view.trial_of("정제된 데이터")

    report.block(
        "무엇이 무너졌는가",
        f"{'':<14}{'accuracy':>10}{'macroF1':>10}{'재현율':>10}\n"
        f"{'오염':<14}{dirty.accuracy:>10.3f}{dirty.macro_f1:>10.3f}"
        f"{dirty.macro_recall:>10.3f}\n"
        f"{'정제':<14}{clean.accuracy:>10.3f}{clean.macro_f1:>10.3f}"
        f"{clean.macro_recall:>10.3f}",
    )

    assert abs(dirty.macro_recall - clean.macro_recall) < 0.1
    assert clean.accuracy > dirty.accuracy + 0.05
    report.note(
        "**재현율은 거의 같다.** 오염된 모델도 이상을 놓치지는 않는다. "
        "무너진 것은 정확도와 macro F1 — 정상을 이상이라고 부르고 있다. "
        "현장에서는 이것이 '알람이 너무 많다'로 나타난다 (실습 5-13)."
    )


def test_데이터를_손잡이에_적었기_때문에_소견이_안_뜬다(quality_impact_lab) -> None:
    view = _board(quality_impact_lab)
    codes = [f.code for f in view.findings]

    report.block(
        "소견", "\n".join(f"  {f.describe()}" for f in view.findings) or "  없음"
    )

    assert "EXP_DATA_DIFFERS" not in codes
    report.note(
        "데이터가 서로 다른 실험이다. 그런데 **손잡이에 data 를 적어 두었다.** "
        "적지 않았으면 '모델을 비교한 것이 아니라 데이터를 비교했다'는 "
        "막는 소견이 떴을 것이다 (실습 3-14)."
    )


def test_품질_개선의_근거는_점수가_아니라_이_표다(quality_impact_lab) -> None:
    view = _board(quality_impact_lab)
    clean = view.trial_of("정제된 데이터")
    dirty = view.trial_of("오염된 데이터")

    report.block(
        "정제 작업의 값어치",
        f"  macro F1  {dirty.macro_f1:.3f} → {clean.macro_f1:.3f}  "
        f"({clean.macro_f1 - dirty.macro_f1:+.3f})\n"
        f"  loss      {dirty.loss:.3f} → {clean.loss:.3f}",
    )

    assert view.best_label == "정제된 데이터"
    report.note(
        "라벨을 다시 붙이고 결측을 메우는 일은 지루하고 티가 안 난다. "
        "**이 표가 그 작업의 예산을 지켜 준다.**"
    )
