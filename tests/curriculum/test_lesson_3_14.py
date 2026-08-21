"""실습 3-14 — 실험을 반복하고 비교하라.

    pytest -m lesson_3_14 -s

모델은 한 번 만들고 끝나지 않는다. 스무 번쯤 만든다.
그러면 두 달 뒤에 반드시 이 질문이 온다.

    "그때 그 0.95 짜리, 어떻게 만든 거였죠?"

기록이 없으면 대답은 "다시 해 봐야 압니다"다.

그리고 반복에는 규칙이 하나 있다.

    **한 번에 하나만 바꾼다.**

두 개를 같이 바꾸고 좋아지면 무엇 때문인지 알 수 없다.
그 상태로 다음 실험을 하면 **잘못된 방향으로 스무 번을 간다.**
"""

from __future__ import annotations

import pytest

from application.model.compare_experiments import (
    CompareExperimentsCommand,
    TrialRequest,
)
from domain.model.experiment import (
    ExperimentBoard,
    ExperimentPolicy,
    ExperimentTrial,
    TrialKnobs,
    TrialMetrics,
)
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_3_14

TRIALS = (
    TrialRequest("run-hp-base", "base", {"lr": "3e-3", "hidden": "16-32"}),
    TrialRequest("run-hp-lr-낮춤", "lr 낮춤", {"lr": "3e-4", "hidden": "16-32"}),
    TrialRequest("run-hp-채널-키움", "채널 키움", {"lr": "3e-3", "hidden": "32-64"}),
    TrialRequest(
        "run-hp-둘-한꺼번에", "둘 한꺼번에", {"lr": "3e-4", "hidden": "32-64"}
    ),
)


def _board(hyperparameter_lab, trials=TRIALS, **kwargs):  # noqa: ANN001, ANN003
    return hyperparameter_lab.model.compare_experiments().execute(
        CompareExperimentsCommand(name="학습 설정을 바꿔 보았다", trials=trials, **kwargs)
    )


def test_스무_번의_실험을_한_표로_본다(hyperparameter_lab) -> None:
    report.section("실습 3-14 · 실험을 반복하고 비교하라")

    view = _board(hyperparameter_lab)
    report.block("비교표", view.render())

    assert len(view.trials) == 4
    assert view.best_label is not None
    report.note(
        "표에는 **결과와 조건이 함께** 있다. "
        "정확도만 적어 두면 두 달 뒤에 재현할 수 없다."
    )


def test_학습률_하나가_모델을_못_쓰게_만든다(hyperparameter_lab) -> None:
    view = _board(hyperparameter_lab)
    base = view.trial_of("base")
    slow = view.trial_of("lr 낮춤")

    report.block(
        "학습률만 바꿨을 때",
        f"  base    lr=3e-3  macro F1 {base.macro_f1:.3f}  loss {base.loss:.3f}\n"
        f"  lr 낮춤 lr=3e-4  macro F1 {slow.macro_f1:.3f}  loss {slow.loss:.3f}",
    )

    assert slow.macro_f1 < base.macro_f1 - 0.1
    report.note(
        f"macro F1 이 {base.macro_f1:.3f} 에서 {slow.macro_f1:.3f} 로 떨어졌다. "
        "**정확도는 {:.3f} 로 아직 높다** — 다수 클래스만 찍고 있기 때문이다. "
        "정확도만 보면 이 실패가 안 보인다 (실습 3-9).".format(slow.accuracy)
    )


def test_한꺼번에_두_개를_바꾸면_원인을_모른다(hyperparameter_lab) -> None:
    """이 실습의 본론."""
    view = _board(hyperparameter_lab)
    codes = [f.code for f in view.findings]

    report.block(
        "비교를 믿어도 되는가",
        "\n".join(f"  {f.describe()}" for f in view.findings),
    )

    assert "EXP_MULTIPLE_KNOBS_CHANGED" in codes
    report.note(
        "'둘 한꺼번에'는 학습률과 채널을 같이 바꿨다. "
        "결과가 좋아졌다면 **둘 중 무엇 때문인지 알 수 없고**, "
        "나빠졌다면 어느 쪽을 되돌려야 할지도 알 수 없다."
    )


def test_시드를_안_적으면_운과_실력을_구분할_수_없다() -> None:
    board = ExperimentBoard(name="시드를 안 적었다")
    for label, seed, f1 in (("A", 42, 0.91), ("B", 7, 0.95)):
        board = board.with_trial(
            ExperimentTrial(
                label=label,
                knobs=TrialKnobs({"hidden": "16-32"}),
                metrics=TrialMetrics(
                    accuracy=f1, macro_recall=f1, macro_f1=f1, loss=0.1,
                    evaluated_samples=500,
                ),
                seed=seed,
                data_ref="power",
            )
        )
    findings = ExperimentPolicy().inspect(board)

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(f.code == "EXP_SEEDS_DIFFER" and f.is_blocking for f in findings)
    report.note(
        "B 가 0.04 더 높다. 그런데 시드가 다르다. "
        "**구조가 좋아서인지 운이 좋아서인지 이 표로는 알 수 없다.** "
        "시드를 바꾼 것도 실험이라면 손잡이에 적어야 한다."
    )


def test_데이터를_바꿔_놓고_모델을_비교했다고_하면_안_된다() -> None:
    board = ExperimentBoard(name="데이터가 달랐다")
    for label, data_ref in (("A", "2026-05 수집분"), ("B", "2026-06 수집분")):
        board = board.with_trial(
            ExperimentTrial(
                label=label,
                knobs=TrialKnobs({"hidden": "16-32"}),
                metrics=TrialMetrics(
                    accuracy=0.9, macro_recall=0.9, macro_f1=0.9, loss=0.1,
                    evaluated_samples=500,
                ),
                seed=42,
                data_ref=data_ref,
            )
        )
    findings = ExperimentPolicy().inspect(board)

    assert any(f.code == "EXP_DATA_DIFFERS" and f.is_blocking for f in findings)
    report.note(
        "6월 데이터가 더 깨끗해서 좋아진 것을 '모델을 개선했다'고 보고하는 일은 "
        "**현장에서 실제로 일어난다.** 데이터도 손잡이다."
    )


def test_같은_이름을_두_번_쓸_수_없다() -> None:
    trial = ExperimentTrial(
        label="A",
        knobs=TrialKnobs({"lr": "3e-3"}),
        metrics=TrialMetrics(
            accuracy=0.9, macro_recall=0.9, macro_f1=0.9, loss=0.1
        ),
        seed=42,
        data_ref="power",
    )
    board = ExperimentBoard(name="중복").with_trial(trial)

    with pytest.raises(InvariantViolation, match="이미 있다"):
        board.with_trial(trial)
    report.note(
        "'실험3', '실험3_최종', '실험3_최종_진짜' 는 현장에서 실제로 생긴다. "
        "이름이 겹치면 어느 것이 그 0.95 였는지 영영 못 찾는다."
    )


def test_무엇을_두었는지_없으면_기록이_아니다() -> None:
    with pytest.raises(InvariantViolation, match="재현할 수 없다"):
        TrialKnobs({})
    report.note("손잡이가 비어 있는 시행은 결과만 있는 종이다.")
