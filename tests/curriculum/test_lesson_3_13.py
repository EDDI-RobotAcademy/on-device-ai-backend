"""실습 3-13 — 통계로 잡히는 이상을 AI라고 부르지 마라.

    pytest -m lesson_3_13 -s

AI 프로젝트를 시작하기 전에 반드시 물어야 하는 질문이 있다.

    **"이거, 통계로는 안 되나?"**

3-시그마 규칙은 학습이 없다. GPU 도 없다. 왜 그렇게 판단했는지 한 줄로 설명된다.
그것으로 끝나는 문제에 신경망을 얹으면 **유지비만 평생 늘어난다.**

그래서 여기서는 통계 기준선을 같은 데이터·같은 창·같은 혼동 행렬로 재서
학습 모델과 나란히 놓는다. 그리고 두 가지를 확인한다.

    1. 통계는 재현율과 정밀도를 **동시에 가지지 못한다**
    2. 통계는 "이상하다"까지만 말한다 — **유형은 말하지 못한다**

두 번째가 AI 를 쓸 진짜 근거다. 정확도가 아니라.
"""

from __future__ import annotations

import pytest

from application.model.compare_with_baseline import CompareWithBaselineCommand
from domain.model.statistical_baseline import (
    BaselineJustificationPolicy,
    DetectionMethod,
    DetectorSpec,
)
from tests.support import report

pytestmark = pytest.mark.lesson_3_13


def _compare(trained_disjoint, *, method=DetectionMethod.THREE_SIGMA, k=3.0, ratio=0.2, **kwargs):  # noqa: ANN001, ANN003
    return trained_disjoint.model.compare_with_baseline().execute(
        CompareWithBaselineCommand(
            run_id=trained_disjoint.run_id,
            detector=DetectorSpec(
                method=method, threshold=k, min_flagged_ratio=ratio
            ),
            **kwargs,
        )
    )


def test_학습_없는_검출기부터_재_본다(trained_disjoint) -> None:
    report.section("실습 3-13 · 통계로 잡히는 이상을 AI라고 부르지 마라")

    view = _compare(trained_disjoint)
    report.block("통계 기준선 대 학습 모델", view.render())

    assert 0.0 <= view.statistical_recall <= 1.0
    report.note(
        "이 검출기에는 epoch 도 loss 도 없다. "
        "train 분할에서 평균과 표준편차를 뽑은 것이 전부다. "
        "**그런데도 숫자가 나온다** — 그 숫자가 기준선이다."
    )


def test_기준을_낮추면_더_잡지만_헛알람이_는다(trained_disjoint) -> None:
    """이 실습의 본론 (1)."""
    rows = []
    for k in (1.0, 1.5, 2.0, 3.0):
        view = _compare(trained_disjoint, k=k, ratio=0.1)
        rows.append((k, view.statistical_recall, view.statistical_precision))

    report.block(
        "3-시그마의 기준을 움직여 보면",
        f"{'k':>6}{'재현율':>10}{'정밀도':>10}\n"
        + "\n".join(f"{k:>6.1f}{r:>10.3f}{p:>10.3f}" for k, r, p in rows),
    )

    loose = rows[0]
    tight = rows[-1]
    assert loose[1] > tight[1], "기준을 낮추면 더 많이 잡아야 한다"
    assert loose[2] < tight[2], "그 대가로 정밀도가 떨어져야 한다"
    report.note(
        f"k=1.0 이면 {loose[1]:.0%} 를 다 잡지만 정밀도가 {loose[2]:.3f} 다 — "
        "**알람 5건 중 4건이 헛것이다.** "
        "현장은 그런 알람을 일주일이면 꺼 버린다. "
        f"k=3.0 으로 조이면 헛알람은 사라지지만 {tight[1]:.0%} 밖에 못 잡는다. "
        "**통계 기준선은 이 둘을 동시에 가지지 못한다.**"
    )


def test_헛알람이_절반을_넘으면_소견이_뜬다(trained_disjoint) -> None:
    view = _compare(trained_disjoint, k=1.0, ratio=0.1)
    codes = [f.code for f in view.findings]

    report.block("소견", "\n".join(f"  {f.describe()}" for f in view.findings))
    assert "BASELINE_TOO_NOISY" in codes
    report.note(
        "재현율만 보고 '통계로 충분하다'고 결론내는 일이 실제로 일어난다. "
        "**정밀도를 같이 봐야 그 결론이 무너진다.**"
    )


def test_통계는_이상의_유형을_말하지_못한다(trained_disjoint) -> None:
    """이 실습의 본론 (2)."""
    view = _compare(trained_disjoint)

    report.block(
        "통계가 채울 수 없는 칸",
        f"  이상 여부만  : 통계 재현율 {view.statistical_recall:.3f} / "
        f"모델 재현율 {view.model_recall:.3f}\n"
        f"  유형까지     : 통계 —      / 모델 {view.model_type_accuracy:.3f} "
        f"(유형 {view.type_count}종)",
    )

    assert view.type_count >= 2
    assert view.model_type_accuracy > 0.0
    assert "BASELINE_CANNOT_TYPE" in [f.code for f in view.findings]
    report.note(
        "FAULT 와 OVERLOAD 는 **대응이 다르다.** "
        "하나는 설비를 세워야 하고 하나는 부하를 낮추면 된다. "
        "'평소와 다르다'만 알려 주는 알람으로는 사람이 뛰어가서 눈으로 봐야 한다. "
        "**AI 를 쓸 근거는 정확도가 아니라 여기에 있다.**"
    )


def test_이기지_못하면_통계로_충분하다() -> None:
    """모델이 기준선을 못 이긴 상황을 직접 만들어 본다."""
    from domain.model.statistical_baseline import BaselineComparison

    comparison = BaselineComparison(
        detector="THREE_SIGMA(k=3)",
        statistical_recall=0.92,
        statistical_precision=0.88,
        model_recall=0.94,
        model_precision=0.89,
        model_type_accuracy=0.0,
        type_count=1,
    )
    findings = BaselineJustificationPolicy(min_recall_gain=0.05).inspect(comparison)

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(f.code == "BASELINE_NOT_BEATEN" and f.is_blocking for f in findings)
    report.note(
        "재현율을 0.02 이겼다. 유형도 하나뿐이라 AI 만 할 수 있는 일이 없다. "
        "**이건 3-시그마 한 줄로 끝낼 문제다.** "
        "여기서 멈추는 판단이 프로젝트를 살린다."
    )


def test_세_가지_규칙은_가정이_서로_다르다(trained_disjoint) -> None:
    rows = []
    for method, k in (
        (DetectionMethod.THREE_SIGMA, 1.5),
        (DetectionMethod.IQR, 1.5),
        (DetectionMethod.EWMA, 1.0),
    ):
        view = _compare(trained_disjoint, method=method, k=k, ratio=0.1)
        rows.append((method.value, view.statistical_recall, view.statistical_precision))

    report.block(
        "규칙마다 다른 것을 본다",
        f"{'규칙':<14}{'재현율':>10}{'정밀도':>10}\n"
        + "\n".join(f"{n:<14}{r:>10.3f}{p:>10.3f}" for n, r, p in rows),
    )

    assert len({(r, p) for _, r, p in rows}) > 1, "규칙이 다르면 결과도 달라야 한다"
    report.note(
        "3-시그마는 정규분포를 가정한다. IQR 은 분포 모양을 가정하지 않는다. "
        "EWMA 는 **기준선이 천천히 따라 움직인다** — "
        "그래서 계절 변화는 흡수하지만 천천히 나빠지는 고장은 놓친다 "
        "(실습 5-7 에서 다시 만나는 함정이다)."
    )
