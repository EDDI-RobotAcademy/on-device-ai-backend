"""실습 3-7 — AI가 데이터를 외우기 시작하는 순간을 잡아라.

    pytest -m lesson_3_7 -s

학습 손실은 계속 떨어지는데 검증 손실이 올라가기 시작하는 지점.
그 지점이 **모델이 패턴 대신 표본을 외우기 시작한 순간**이다.

그리고 중요한 것: 저장해야 할 모델은 **마지막이 아니라 그 지점의 것**이다.
"""

from __future__ import annotations

import pytest

from domain.model.curve import EpochRecord, OverfittingPolicy, TrainingCurve
from tests.support import report

pytestmark = pytest.mark.lesson_3_7


def curve_of(pairs: list[tuple[float, float, float, float]]) -> TrainingCurve:
    return TrainingCurve(
        records=tuple(
            EpochRecord(i + 1, tl, vl, ta, va)
            for i, (tl, vl, ta, va) in enumerate(pairs)
        )
    )


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_외우기_시작한_지점을_찾는다() -> None:
    report.section("실습 3-7 · AI가 데이터를 외우기 시작하는 순간을 잡아라")

    curve = curve_of(
        [
            (1.10, 1.05, 0.40, 0.42),
            (0.60, 0.55, 0.75, 0.76),
            (0.35, 0.38, 0.88, 0.87),
            (0.20, 0.34, 0.93, 0.89),  # ← 최저점
            (0.12, 0.39, 0.96, 0.88),
            (0.06, 0.47, 0.99, 0.87),
            (0.02, 0.58, 1.00, 0.86),
        ]
    )
    report.block("전형적인 과적합 곡선", curve.render())

    assert curve.best_epoch.epoch == 4
    assert curve.overfitting_epoch == 5

    findings = OverfittingPolicy().inspect(curve)
    assert "OVERFIT_ONSET" in codes(findings)
    assert "OVERFIT_LOSS_RISING" in codes(findings)
    assert "OVERFIT_GAP_WIDE" in codes(findings)

    report.note(
        "epoch 5 부터 검증 손실이 올라간다. 학습 손실은 계속 떨어진다. "
        "모델은 계속 '좋아지고' 있는데, 새 데이터에 대해서는 나빠지고 있다."
    )
    report.note(
        f"마지막 epoch 의 일반화 격차는 {curve.final_gap:+.2f} — "
        "학습 100%, 검증 86%. 14%p 만큼을 외운 것이다."
    )


def test_저장해야_할_모델은_마지막이_아니다() -> None:
    curve = curve_of(
        [(1.0, 0.9, 0.4, 0.45), (0.4, 0.30, 0.85, 0.90), (0.1, 0.55, 0.99, 0.84)]
    )
    best, last = curve.best_epoch, curve.last

    report.block(
        "어느 가중치를 저장할 것인가",
        f"  epoch {best.epoch} : val_loss {best.validation_loss:.2f} "
        f"val_acc {best.validation_accuracy:.2f}   ← 이것\n"
        f"  epoch {last.epoch} : val_loss {last.validation_loss:.2f} "
        f"val_acc {last.validation_accuracy:.2f}   (마지막)",
    )
    assert best.epoch == 2
    assert best.validation_accuracy > last.validation_accuracy
    report.note(
        "마지막 가중치를 그대로 배포하면 최고 성능을 6%p 버리는 것이다. "
        "이 프로젝트의 학습기는 최저점의 가중치를 따로 보관한다."
    )


def test_우리_학습기는_최저점_가중치를_쓴다(trained) -> None:
    """평가할 때 마지막이 아니라 best_state 를 불러온다."""
    view = trained.curve
    report.block("우리 모델의 곡선", view.render())

    if view.overfitting_epoch:
        report.note(
            f"epoch {view.overfitting_epoch} 부터 검증 손실이 올라갔다. "
            f"평가에 쓰인 것은 epoch {view.best_epoch} 의 가중치다."
        )
    assert view.best_epoch is not None


def test_조기_종료는_외우는_시간을_벌어_주지_않는_장치다(trained_early_stop) -> None:
    view = trained_early_stop.curve
    report.block("조기 종료를 건 학습", view.render())

    assert view.epoch_count < 30
    report.note(
        f"30 epoch 을 요청했지만 {view.epoch_count} epoch 에서 멈췄다. "
        "검증 손실이 좋아지지 않는데 계속 도는 것은 시간 낭비가 아니라 "
        "**외우는 시간을 벌어 주는 것**이다."
    )


def test_격차가_벌어지지_않으면_경고하지_않는다() -> None:
    healthy = curve_of(
        [(1.0, 1.0, 0.4, 0.41), (0.5, 0.48, 0.8, 0.79), (0.3, 0.29, 0.9, 0.88)]
    )
    assert healthy.overfitting_epoch is None
    assert OverfittingPolicy().inspect(healthy) == ()
    report.note("검증 손실이 계속 떨어지는 동안은 아무 말도 하지 않는다.")
