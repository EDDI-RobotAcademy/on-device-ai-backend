"""실습 5-6 — Prediction Distribution의 변화를 추적하라.

    pytest -m lesson_5_6 -s

**이 지표의 값어치는 하나다 — 정답이 없어도 잴 수 있다.**

정확도는 못 잰다. 그런데 모델이 무엇을 얼마나 답하고 있는지는 셀 수 있다.
"""

from __future__ import annotations

import pytest

from domain.operations.prediction_mix import PredictionDriftPolicy, PredictionMix
from domain.operations.window import ObservationWindow
from domain.shared.inspection import Severity
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_6

BASELINE = {"NORMAL": 0.78, "OVERLOAD": 0.21, "FAULT": 0.01}


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def window(label: str = "w") -> ObservationWindow:
    return ObservationWindow(
        label=label,
        started_at="2026-05-23 00:00:00",
        ended_at="2026-05-23 07:59:59",
        sample_count=3000,
    )


def test_평가_때의_기준과_현장은_처음부터_다르다(deployed) -> None:
    """그래서 기준을 다시 잡는다."""
    report.section("실습 5-6 · Prediction Distribution의 변화를 추적하라")

    from domain.operations.identifiers import WatchId

    watch = deployed.operations.watches.find_by_id(WatchId.of(deployed.watch_id))
    report.block(
        "기준 예측 분포 (현장 1일차로 재고정한 뒤)",
        "\n".join(
            f"  {label:<10}{share:>8.1%}"
            for label, share in sorted(
                watch.baseline_mix.items(), key=lambda x: -x[1]
            )
        ),
    )
    report.note(
        "처음 기준은 모듈 4 의 **평가 데이터**에서 왔다 (FAULT 4.5%). "
        "그런데 현장 1일차의 FAULT 는 0.8% 다."
    )
    report.note(
        "평가 집합에는 사건이 골고루 들어 있고(실습 3-8), 현장은 대개 평온하기 때문이다. "
        "이 상태로 두면 **처음부터** 어긋나 있어서 진짜 변화가 묻힌다."
    )
    report.note(
        "그래서 배포 직후 아무 일도 없었던 창을 새 기준으로 못박는다. "
        "단, 그 구간이 정말 평온했는지는 사람이 확인해야 한다."
    )
    assert watch.baseline_mix


def test_알람이_몇_배로_늘었는지_센다(deployed) -> None:
    first, last = deployed.reports[0], deployed.reports[-1]
    surge = [f for f in last.findings if f.code == "OPS_LABEL_SURGE"]

    report.block(
        "1일차 vs 4일차 마지막 창",
        "\n".join(
            [
                f"  분포 이동 : {first.prediction_shift:.1%} → {last.prediction_shift:.1%}",
                *(f"  {f.describe()}" for f in surge),
            ]
        ),
    )
    assert surge
    report.note(
        "FAULT 예측이 기준의 5배가 됐다. "
        "설비 정지 알람이 하루 5배로 울리고 있다는 뜻이다."
    )
    report.note(
        "**여기서 정확도는 여전히 모른다.** 알 수 있는 것은 '무언가 달라졌다' 까지다. "
        "그것만으로도 사람을 부르기에는 충분하다."
    )


def test_확신도가_올라가도_안심할_수_없다(deployed) -> None:
    """이 실습에서 가장 불편한 사실."""
    first, last = deployed.reports[0], deployed.reports[-1]
    report.block(
        "평균 확신도",
        "\n".join(
            [
                f"  1일차 : {first.confidence:.3f}",
                f"  4일차 : {last.confidence:.3f}",
            ]
        ),
    )
    assert last.confidence >= first.confidence
    report.note(
        "확신도는 오히려 **올라갔다.** 그런데 정답이 붙은 표본으로 재 보면 "
        "정확도는 0.83 → 0.73 으로 떨어졌다."
    )
    report.note(
        "**모델은 자기가 틀린 줄 모른다.** softmax 는 '이 셋 중에 뭐가 제일 큰가' 를 "
        "말할 뿐, '내가 이 입력을 본 적 있는가' 를 말하지 않는다."
    )
    report.note(
        "그래서 확신도 하락은 신호가 되지만, 확신도 유지는 안전의 근거가 못 된다. "
        "입력 분포를 따로 봐야 한다 (실습 5-7)."
    )


def test_분포가_크게_달라지면_둘_중_하나다() -> None:
    shifted = PredictionMix(
        window=window(),
        counts={"NORMAL": 1230, "OVERLOAD": 660, "FAULT": 1110},
        mean_confidence={"NORMAL": 0.9, "OVERLOAD": 0.85, "FAULT": 0.8},
    )
    findings = PredictionDriftPolicy().inspect(shifted, BASELINE)
    report.block("NORMAL 78% → 41%", shifted.render(BASELINE))
    report.block("소견", "\n".join(f"  - {f.describe()}" for f in findings))

    assert "OPS_PREDICTION_SHIFT" in codes(findings)
    report.note(
        "가능성은 둘이다. **세상이 변했거나**(설비가 실제로 자주 선다) "
        "**모델이 무너졌거나**(입력이 변해 엉뚱한 답을 한다)."
    )
    report.note("둘 다 사람이 확인해야 하는 상황이다. 그게 이 소견의 목적이다.")


def test_한_클래스가_사라지면_그_클래스에_모델은_없는_것이다() -> None:
    vanished = PredictionMix(
        window=window(),
        counts={"NORMAL": 2400, "OVERLOAD": 600},
        mean_confidence={"NORMAL": 0.95, "OVERLOAD": 0.9},
    )
    findings = PredictionDriftPolicy(
        critical_labels=frozenset({"FAULT"})
    ).inspect(vanished, BASELINE)

    blocking = [f for f in findings if f.severity is Severity.CRITICAL]
    assert "OPS_LABEL_VANISHED" in codes(findings)
    assert blocking
    report.note(
        "실습 3-9 의 EVAL_CLASS_NEVER_PREDICTED 가 **현장에서 다시 나타난 것**이다. "
        "평가에서는 잡았는데 현장에서 또 이렇게 되는 일이 있다."
    )
    report.note(
        "놓치면 안 되는 클래스(FAULT)면 CRITICAL, 아니면 WARNING 이다. "
        "그 구분은 현장이 정한다."
    )


def test_확신도가_내려가면_모델이_헷갈리는_것이다() -> None:
    unsure = PredictionMix(
        window=window(),
        counts={"NORMAL": 2340, "OVERLOAD": 630, "FAULT": 30},
        mean_confidence={"NORMAL": 0.52, "OVERLOAD": 0.48, "FAULT": 0.41},
    )
    findings = PredictionDriftPolicy().inspect(
        unsure, BASELINE, baseline_confidence=0.94
    )
    found = codes(findings)

    assert "OPS_LOW_CONFIDENCE" in found
    assert "OPS_CONFIDENCE_DROP" in found
    report.note(
        "분포는 그대로인데 확신도만 내려갔다. "
        "**답은 같은데 자신이 없어진 것** — 입력이 학습 때와 달라졌다는 신호다."
    )


def test_기준이_없으면_달라졌는지_말할_수_없다() -> None:
    findings = PredictionDriftPolicy().inspect(
        PredictionMix(window=window(), counts={"NORMAL": 100}), {}
    )
    assert "OPS_NO_MIX_BASELINE" in codes(findings)
    report.note("현장 분포만 봐서는 그것이 정상인지 이상인지 알 수 없다.")


def test_기준_재고정은_이유_없이_할_수_없다(operations_container) -> None:
    """**판정 기준을 바꾸는 일이다.**"""
    from domain.shared.errors import InvariantViolation

    windows = os5.windows(operations_container)
    with pytest.raises(InvariantViolation) as caught:
        os5.rebaseline(operations_container, windows[0], reason="")
    report.note(str(caught.value))
    report.note(
        "이상한 구간을 기준으로 잡으면 **그 이상이 정상이 된다.** "
        "그래서 이유가 기록으로 남아야 한다."
    )
