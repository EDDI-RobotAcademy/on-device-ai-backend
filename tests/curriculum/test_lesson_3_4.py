"""실습 3-4 — 센서의 시간을 AI가 볼 수 있게 잘라라.

    pytest -m lesson_3_4 -s

실습 1-7 에서 "몇 초를 한 덩어리로 볼 것인가"를 정했다.
여기서는 그 창을 실제로 자를 때 생기는 두 문제를 다룬다.

    1. 창의 라벨은 무엇인가?
    2. 창이 겹치면 표본이 몇 개인가? (그리고 분할은 어떻게 되는가)
"""

from __future__ import annotations

import pytest

from domain.model.windowing import (
    WindowingPlan,
    WindowingPolicy,
    WindowingSummary,
    WindowLabelPolicy,
)
from domain.shared.errors import InvariantViolation
from tests.support import model_scenario as ms
from tests.support import report

pytestmark = pytest.mark.lesson_3_4


def test_창의_라벨은_현장이_정한다() -> None:
    report.section("실습 3-4 · 센서의 시간을 AI가 볼 수 있게 잘라라")

    counts = {"NORMAL": 21, "FAULT": 9}  # 30 표본 중 9 표본이 트립

    다수결 = WindowLabelPolicy(
        priority=(("FAULT", 0.51), ("OVERLOAD", 0.51)), default_label="NORMAL"
    )
    현장규칙 = ms.window_label_policy()

    report.block(
        "같은 창, 다른 규칙",
        f"  창 안의 라벨 : {counts}\n"
        f"  다수결       : {다수결.label_for(counts)}\n"
        f"  현장 규칙    : {현장규칙.label_for(counts)}   ({현장규칙.describe()})",
    )
    assert 다수결.label_for(counts) == "NORMAL"
    assert 현장규칙.label_for(counts) == "FAULT"
    report.note(
        "다수결로 하면 5분 중 1분 반 동안 설비가 멈춰 있던 창이 '정상'이 된다. "
        "짧게 스쳐도 사고는 사고다 — 그 판단이 30% 라는 숫자에 들어 있다."
    )


def test_규칙에는_기준이_있어야_한다() -> None:
    with pytest.raises(InvariantViolation, match="비어 있다"):
        WindowLabelPolicy(priority=(), default_label="NORMAL")
    with pytest.raises(InvariantViolation, match="0 초과 1 이하"):
        WindowLabelPolicy(priority=(("FAULT", 1.5),), default_label="NORMAL")


def test_창_길이는_모델_입력과_같아야_한다() -> None:
    """다르면 학습은 그냥 돌아가고, 배포할 때 터진다."""
    with pytest.raises(InvariantViolation, match="통째로 버려진다"):
        WindowingPlan(
            window_length=30, stride=60, label_policy=ms.window_label_policy()
        )

    plan = ms.windowing_plan(stride=10)
    assert plan.overlap_ratio == pytest.approx(2 / 3)


def test_창이_겹치면_표본_수가_부풀려진다(trained) -> None:
    report.block("실제로 자른 결과", trained.preparation.windowing_report)

    codes = {f.code for f in trained.preparation.findings}
    assert "WINDOW_OVERLAP_HIGH" in codes
    report.note(
        "창 1,294개처럼 보이지만 독립 표본은 431개다. "
        "겹친 창끼리는 같은 원본 표본을 공유하기 때문이다."
    )
    report.note("'데이터가 1,294개 있습니다'라는 보고가 3배 부풀려진 것이다.")


def test_stride_를_바꾸면_표본_수와_겹침이_함께_바뀐다() -> None:
    dense = WindowingSummary(
        source_row_count=12_960,
        window_length=30,
        stride=10,
        window_count=1_294,
        label_counts={"NORMAL": 1008, "OVERLOAD": 236, "FAULT": 50},
    )
    sparse = WindowingSummary(
        source_row_count=12_960,
        window_length=30,
        stride=30,
        window_count=432,
        label_counts={"NORMAL": 336, "OVERLOAD": 79, "FAULT": 17},
    )

    report.block(
        "stride 10 vs 30",
        f"{'':10}{'창':>8}{'겹침':>8}{'독립표본':>10}{'FAULT창':>9}\n"
        f"{'stride 10':10}{dense.window_count:>8,}{dense.overlap_ratio:>8.0%}"
        f"{dense.effective_sample_count:>10,}{dense.label_counts['FAULT']:>9}\n"
        f"{'stride 30':10}{sparse.window_count:>8,}{sparse.overlap_ratio:>8.0%}"
        f"{sparse.effective_sample_count:>10,}{sparse.label_counts['FAULT']:>9}",
    )
    assert dense.effective_sample_count == pytest.approx(
        sparse.window_count, abs=5
    )
    report.note(
        "겹침을 없애면 표본이 3분의 1로 준다. 대신 그 표본은 전부 서로 다르다. "
        "**어느 쪽이 정직한 숫자인가.**"
    )


def test_stride_가_너무_크면_데이터가_버려진다() -> None:
    summary = WindowingSummary(
        source_row_count=1_000, window_length=30, stride=100, window_count=10
    )
    findings = WindowingPolicy().inspect(summary)
    codes = {f.code for f in findings}
    assert "WINDOW_COVERAGE_LOW" in codes
    report.note("stride 100, 창 30 — 표본 70개마다 한 번씩 통째로 건너뛴다.")


def test_원본에_있던_클래스가_창에서_사라질_수_있다() -> None:
    """짧은 사건은 창으로 자르는 과정에서 소멸한다."""
    summary = WindowingSummary(
        source_row_count=12_960,
        window_length=30,
        stride=30,
        window_count=432,
        label_counts={"NORMAL": 420, "OVERLOAD": 9, "FAULT": 3},
    )
    findings = WindowingPolicy(min_windows_per_class=10).inspect(summary)
    subjects = {f.subject for f in findings if f.code == "WINDOW_CLASS_TOO_FEW"}

    assert subjects == {"OVERLOAD", "FAULT"}
    report.note(
        "원본에는 FAULT 표본이 240개 있었는데 창은 3개뿐이다. "
        "표본 수를 원본 기준으로 세면 이 사실을 놓친다."
    )
