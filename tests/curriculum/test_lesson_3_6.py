"""실습 3-6 — Loss가 떨어지는 순간을 확인하라.

    pytest -m lesson_3_6 -s

Loss 숫자 하나만 보면 아무것도 알 수 없다. **곡선의 모양**을 봐야 한다.
그리고 그 판단은 사람이 그래프를 눈으로 보고 하는 일이었다.
눈으로 하면 사람마다 다르고, 기록에 남지 않는다. 그래서 규칙으로 만든다.
"""

from __future__ import annotations

import pytest

from domain.model.curve import EpochRecord, LearningPolicy, TrainingCurve
from tests.support import report

pytestmark = pytest.mark.lesson_3_6


def curve_of(pairs: list[tuple[float, float, float, float]]) -> TrainingCurve:
    return TrainingCurve(
        records=tuple(
            EpochRecord(i + 1, tl, vl, ta, va)
            for i, (tl, vl, ta, va) in enumerate(pairs)
        )
    )


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_학습이_실제로_일어났는지_곡선으로_확인한다(trained) -> None:
    report.section("실습 3-6 · Loss가 떨어지는 순간을 확인하라")

    view = trained.curve
    report.block("학습 곡선", view.render())

    assert view.train_loss_drop > 0.5
    assert view.best_epoch is not None
    report.note(
        f"학습 손실이 {view.train_loss_drop:.1%} 떨어졌다. "
        "이 숫자가 0 에 가까우면 모델은 아무것도 배우지 않은 것이다."
    )


def test_Loss_가_안_떨어지면_원인은_대개_셋이다() -> None:
    flat = curve_of([(1.10, 1.10, 0.33, 0.33)] * 8)
    findings = LearningPolicy().inspect(flat)

    assert "LEARNING_LOSS_FLAT" in codes(findings)
    report.block(
        "배우지 못하는 곡선",
        f"  8 epoch 동안 손실 1.10 그대로\n"
        f"  → {[f.message for f in findings if f.code == 'LEARNING_LOSS_FLAT'][0]}",
    )
    report.note(
        "learning rate 가 너무 작거나 크다 / 입력이 정규화되지 않았다 / "
        "라벨이 입력과 연결되지 않았다 — 이 셋을 먼저 본다."
    )


def test_손실이_떨어져도_baseline_을_못_넘으면_배운_것이_아니다() -> None:
    """불균형 데이터에서 흔한 함정이다. (실습 2-5 와 이어진다)"""
    lazy = curve_of(
        [(1.10, 1.05, 0.74, 0.74), (0.60, 0.58, 0.74, 0.74), (0.40, 0.42, 0.75, 0.744)]
    )
    findings = LearningPolicy().inspect(lazy, baseline_accuracy=0.744)

    assert "LEARNING_NO_BETTER_THAN_BASELINE" in codes(findings)
    report.note(
        "손실은 1.10 → 0.40 으로 잘 떨어졌다. "
        "그런데 정확도는 다수 클래스만 찍는 74.4% 그대로다. "
        "모델은 '전부 NORMAL' 이라고 말하는 법을 배웠을 뿐이다."
    )


def test_우리_모델은_baseline_을_넘었는가(trained) -> None:
    evaluation = trained.evaluation
    report.block(
        "정확도 vs baseline",
        f"  모델 정확도 : {evaluation.accuracy:.3f}\n"
        f"  baseline    : {evaluation.baseline_accuracy:.3f}\n"
        f"  실제로 번 것: {evaluation.accuracy - evaluation.baseline_accuracy:+.3f}",
    )
    assert evaluation.accuracy > evaluation.baseline_accuracy + 0.05


def test_최저점_이후로_더_돈_횟수를_센다(trained) -> None:
    view = trained.curve
    report.note(
        f"최저점 epoch {view.best_epoch} 이후로 {view.wasted_epochs} epoch 을 더 돌았다. "
        "조기 종료를 걸면 그만큼의 시간과 과적합을 아낀다."
    )
    assert view.wasted_epochs >= 0

    wasteful = curve_of(
        [(1.0, 0.9, 0.4, 0.5), (0.5, 0.4, 0.8, 0.85)]
        + [(0.3 - i * 0.01, 0.45 + i * 0.01, 0.9, 0.84) for i in range(8)]
    )
    findings = LearningPolicy(max_epochs_without_progress=5).inspect(wasteful)
    assert "LEARNING_WASTED_EPOCHS" in codes(findings)


def test_epoch_이_없으면_판단할_것도_없다() -> None:
    findings = LearningPolicy().inspect(TrainingCurve())
    assert codes(findings) == {"LEARNING_NO_EPOCH"}


def test_곡선의_판단은_파일_없이_이루어진다() -> None:
    """Domain 테스트는 pandas 도 torch 도 부르지 않는다."""
    curve = curve_of([(1.0, 1.0, 0.4, 0.4), (0.2, 0.25, 0.9, 0.88)])

    assert curve.train_loss_drop == pytest.approx(0.8)
    assert curve.best_epoch.epoch == 2
    assert len(curve) == 2
    report.note("숫자만 있으면 판단할 수 있다. 그래서 이 테스트는 0.001초에 끝난다.")
