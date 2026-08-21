"""실습 5-14 — 트래픽을 나눠서 두 모델을 동시에 재라.

    pytest -m lesson_5_14 -s

실습 5-9 의 그림자(shadow)와 다르다. 그 차이가 이 실습의 전부다.

    그림자   새 모델도 돌리지만 **답은 안 쓴다.** 안전하다. 대신 현장 결과가 없다.
    A/B      트래픽의 일부에 **실제로 새 모델의 답을 쓴다.** 현장 결과가 나온다.
             대신 그 일부는 진짜 위험을 진다.

그래서 A/B 에는 그림자에 없는 규율이 세 개 필요하다.

    나누는 기준을 **고정**한다 — 매번 무작위면 비교가 성립하지 않는다
    양쪽에 **충분한 표본과 디바이스**가 있어야 한다
    **멈출 기준을 시작 전에 숫자로 적어 둔다**

셋째가 가장 중요하다. "나빠 보이면 멈춘다"는 기준이 아니다.
"""

from __future__ import annotations

import pytest

from domain.operations.experiment_split import (
    ArmResult,
    SplitOutcome,
    SplitPolicy,
    TrafficSplit,
)
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_5_14


def _outcome(**overrides):  # noqa: ANN003, ANN202
    control = ArmResult(
        version="v1.2.0",
        sample_count=9_000,
        device_count=9,
        latency_ms_p95=24.0,
        alert_count=180,
        confirmed_true=96,
        confirmed_false=64,
    )
    defaults = dict(
        version="v1.3.0",
        sample_count=1_000,
        device_count=1,
        latency_ms_p95=26.0,
        alert_count=14,
        confirmed_true=11,
        confirmed_false=3,
    )
    defaults.update(overrides)
    candidate = ArmResult(**defaults)
    return SplitOutcome(
        split=TrafficSplit(
            control_version="v1.2.0",
            candidate_version="v1.3.0",
            candidate_ratio=0.1,
        ),
        control=control,
        candidate=candidate,
    )


def test_그림자와_달리_현장_결과가_나온다() -> None:
    report.section("실습 5-14 · 트래픽을 나눠서 두 모델을 동시에 재라")

    outcome = _outcome(
        version="v1.3.0", sample_count=9_500, device_count=9,
        latency_ms_p95=25.0, alert_count=95,
        confirmed_true=78, confirmed_false=12,
    )
    report.block("A/B", outcome.render())

    assert outcome.precision_gain > 0
    report.note(
        "정밀도가 올랐고 알람 건수가 줄었다. "
        "**그림자에서는 이 숫자를 만들 수 없다** — "
        "사람이 확인해 준 정답은 실제로 답을 쓴 쪽에만 붙기 때문이다."
    )


def test_한_대에서만_돌린_비교는_비교가_아니다() -> None:
    outcome = _outcome()
    findings = SplitPolicy().inspect(outcome)

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    codes = {f.code for f in findings}

    assert "AB_TOO_FEW_DEVICES" in codes
    report.note(
        "후보가 디바이스 한 대에서만 돌았다. "
        "**그 설비의 특성이 곧 결과가 된다** — "
        "그 설비가 원래 문제가 많았다면 새 모델이 나빠 보이고, "
        "원래 조용했다면 좋아 보인다."
    )


def test_정답이_없으면_더_낫다를_말할_수_없다() -> None:
    outcome = _outcome(
        version="v1.3.0", sample_count=9_000, device_count=9,
        confirmed_true=3, confirmed_false=1,
    )
    findings = SplitPolicy().inspect(outcome)

    assert any(f.code == "AB_NO_GROUND_TRUTH" and f.is_blocking for f in findings)
    report.note(
        "예측 분포가 달라진 것은 **어느 쪽이 옳은지 말해 주지 않는다.** "
        "현장에서 정답이 붙는 비율은 12% 정도다 (O06) — "
        "그래서 A/B 는 정답을 모으는 절차와 함께 설계해야 한다."
    )


def test_멈출_기준은_시작_전에_숫자로_적어_둔다() -> None:
    """이 실습의 본론."""
    outcome = _outcome(
        version="v1.3.0", sample_count=9_000, device_count=9,
        latency_ms_p95=26.0, alert_count=300,
        confirmed_true=60, confirmed_false=140,
    )
    findings = SplitPolicy(stop_on_precision_drop=0.1).inspect(outcome)

    report.block(
        "멈춤 판정",
        f"  대조군 정밀도 {outcome.control.precision:.3f}\n"
        f"  후보 정밀도   {outcome.candidate.precision:.3f}\n"
        f"  차이          {outcome.precision_gain:+.3f}\n\n"
        + "\n".join(f"  {f.describe()}" for f in findings),
    )

    assert any(f.code == "AB_STOP_PRECISION" and f.is_blocking for f in findings)
    report.note(
        "**이 기준을 시작 전에 적어 두었기 때문에 지금 논쟁이 필요 없다.** "
        "적어 두지 않으면 '조금 더 지켜보자'가 반복되고, "
        "그 사이 현장은 헛알람을 계속 받는다."
    )


def test_느려진_것만으로도_멈춘다() -> None:
    outcome = _outcome(
        version="v1.3.0", sample_count=9_000, device_count=9,
        latency_ms_p95=48.0, alert_count=90,
        confirmed_true=80, confirmed_false=10,
    )
    findings = SplitPolicy().inspect(outcome)

    assert any(f.code == "AB_STOP_LATENCY" and f.is_blocking for f in findings)
    report.note(
        "정밀도는 올랐다. **그런데 p95 가 두 배다.** "
        "사이클을 놓치는 모델은 정확해도 못 쓴다 (실습 4-14)."
    )


def test_실제_배분이_설계와_어긋나면_공평하지_않다() -> None:
    outcome = _outcome(
        version="v1.3.0", sample_count=6_000, device_count=6,
        confirmed_true=60, confirmed_false=20,
    )
    findings = SplitPolicy().inspect(outcome)

    report.block(
        "배분",
        f"  설계 {outcome.split.candidate_ratio:.0%} / "
        f"실제 {outcome.actual_ratio:.1%}",
    )
    assert any(f.code == "AB_ASSIGNMENT_SKEWED" for f in findings)
    report.note(
        "10%로 나눴는데 실제는 40%다. "
        "**나누기가 고장났거나 한쪽 디바이스가 더 바쁘다** — "
        "어느 쪽이든 지금 비교는 공평하지 않다."
    )


def test_나누는_기준이_없으면_비교가_성립하지_않는다() -> None:
    with pytest.raises(InvariantViolation, match="나누는 기준이 없다"):
        TrafficSplit(
            control_version="v1", candidate_version="v2", assignment_key="  "
        )
    report.note(
        "요청마다 무작위로 나누면 같은 설비가 어제는 A, 오늘은 B 가 된다. "
        "**차이가 모델 때문인지 설비 때문인지 영영 알 수 없다.**"
    )


def test_같은_버전을_AB_로_나눌_수_없다() -> None:
    with pytest.raises(InvariantViolation, match="같은 버전"):
        TrafficSplit(control_version="v1.2.0", candidate_version="v1.2.0")
