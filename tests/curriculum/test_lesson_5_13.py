"""실습 5-13 — 알람이 너무 많으면 아무도 안 본다.

    pytest -m lesson_5_13 -s

모델이 이상을 잡았다고 알람이 되는 것이 아니다.
**알람은 사람이 반응할 수 있을 때만 알람이다.**

현장에서 실제로 벌어지는 일:

    1주차  알람이 뜬다. 사람이 달려간다.
    2주차  하루 200번 뜬다. 몇 개만 본다.
    3주차  소리를 끈다.
    4주차  진짜 사고가 났을 때 아무도 못 봤다.

그래서 알람에는 규율이 필요하다 — 확인, 억제, 상한.
여기서 정하는 것은 "무엇이 이상인가"가 아니다. 그건 모델이 한다.
**"그것을 어떻게 알릴 것인가"**다.
"""

from __future__ import annotations

import pytest

from domain.operations.alerting import (
    AlertFatiguePolicy,
    AlertGate,
    AlertRule,
    Signal,
)
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_5_13


def _signals(count: int, *, every: float = 30.0, confidence: float = 0.95):  # noqa: ANN202
    return [
        Signal(at_seconds=i * every, label="FAULT", confidence=confidence)
        for i in range(count)
    ]


def test_규율을_적용한_결과를_원본과_함께_본다(device_pipeline) -> None:
    report.section("실습 5-13 · 알람이 너무 많으면 아무도 안 본다")

    ledger = device_pipeline.ledger
    report.block("알람 규율", ledger.render())

    assert ledger.raw_signal_count > ledger.alert_count
    report.note(
        f"모델은 {ledger.raw_signal_count:,}번 '이상'이라고 했다. "
        f"실제로 사람을 부른 것은 {ledger.alert_count:,}번이다. "
        "**둘 중 하나만 보면 '잘 잡는다'와 '너무 많다'를 구분할 수 없다.**"
    )


def test_규율이_없으면_알람이_쏟아진다(device_pipeline) -> None:
    """이 실습의 본론."""
    signals = [
        Signal(at_seconds=s.at_seconds, label=s.label, confidence=s.confidence)
        for s in _reconstruct(device_pipeline)
    ]
    naive = AlertGate().apply(
        AlertRule(
            alert_labels=("FAULT", "OVERLOAD"),
            dwell=1,
            min_confidence=0.0,
            cooldown_seconds=0.0,
            hourly_budget=100_000,
        ),
        signals,
    )
    disciplined = device_pipeline.ledger

    report.block(
        "규율 전후",
        f"  규율 없음 : {naive.alert_count:>6,}건  "
        f"(시간당 {naive.alerts_per_hour:>7.1f}건)\n"
        f"  규율 있음 : {disciplined.alert_count:>6,}건  "
        f"(시간당 {disciplined.alerts_per_hour:>7.1f}건)",
    )

    assert naive.alert_count > disciplined.alert_count * 5
    report.note(
        f"같은 모델, 같은 데이터다. **규율 하나로 {naive.alert_count:,}건이 "
        f"{disciplined.alert_count:,}건이 되었다.** "
        "잡는 능력이 달라진 것이 아니라 알리는 방식이 달라진 것이다."
    )


def test_한_번_튄_것으로_사람을_부르지_않는다() -> None:
    signals = [
        Signal(at_seconds=0.0, label="NORMAL", confidence=0.99),
        Signal(at_seconds=30.0, label="FAULT", confidence=0.99),  # 한 번 튐
        Signal(at_seconds=60.0, label="NORMAL", confidence=0.99),
    ]
    gate = AlertGate()
    strict = gate.apply(AlertRule(alert_labels=("FAULT",), dwell=3), signals)
    loose = gate.apply(AlertRule(alert_labels=("FAULT",), dwell=1), signals)

    report.block(
        "연속 확인",
        f"  3회 연속 요구 : 알람 {strict.alert_count}건\n"
        f"  1회로 충분    : 알람 {loose.alert_count}건",
    )

    assert strict.alert_count == 0
    assert loose.alert_count == 1
    report.note(
        "**현장 신호는 원래 가끔 튄다.** 센서 하나가 한 표본 흔들린 것으로 "
        "새벽 3시에 사람을 부르면, 다음 주에 그 알람은 꺼져 있다."
    )


def test_확신이_낮으면_알람으로_올리지_않는다() -> None:
    ledger = AlertGate().apply(
        AlertRule(alert_labels=("FAULT",), dwell=1, min_confidence=0.8),
        _signals(5, confidence=0.55),
    )
    assert ledger.alert_count == 0
    assert ledger.withheld_low_confidence == 5
    report.note(
        "확신 0.55 로 '고장입니다'라고 부르지 않는다. "
        "**그 대신 보류로 세어 둔다** — 보류가 많으면 그것도 신호다 (실습 5-12)."
    )


def test_같은_상태가_이어지면_한_번만_낸다() -> None:
    gate = AlertGate()
    ledger = gate.apply(
        AlertRule(alert_labels=("FAULT",), dwell=1, cooldown_seconds=300.0),
        _signals(20, every=30.0),  # 10분 동안 연속 이상
    )

    report.block(
        "억제",
        f"  원본 신호 {ledger.raw_signal_count}건 → 알람 {ledger.alert_count}건\n"
        f"  억제 시간 내 {ledger.suppressed_by_cooldown}건",
    )
    assert ledger.alert_count <= 3
    report.note(
        "10분 동안 계속 고장이다. **사람은 그 사실을 이미 안다.** "
        "30초마다 다시 부르는 것은 정보가 아니라 소음이다."
    )


def test_상한을_넘으면_버리지_않고_묶는다() -> None:
    ledger = AlertGate().apply(
        AlertRule(
            alert_labels=("FAULT",),
            dwell=1,
            cooldown_seconds=0.0,
            hourly_budget=5,
        ),
        _signals(40, every=30.0),  # 20분에 40건
    )

    report.block(
        "상한",
        f"  알람 {ledger.alert_count}건 (묶인 것 포함)\n"
        f"  묶인 건수 {ledger.merged_by_budget}건",
    )

    assert ledger.merged_by_budget > 0
    merged = [a for a in ledger.alerts if a.is_merged]
    assert merged
    report.note(
        "**버리면 그 사건이 없던 일이 된다.** 묶으면 '이 시간에 N건'이 남는다. "
        "그리고 상한에 걸렸다는 사실 자체가 '무언가 크게 잘못되었다'는 신호다."
    )


def test_시간당_알람_수로_현장이_버틸지_판정한다() -> None:
    ledger = AlertGate().apply(
        AlertRule(alert_labels=("FAULT",), dwell=1, cooldown_seconds=0.0),
        _signals(60, every=60.0),  # 1시간에 60건
    )
    findings = AlertFatiguePolicy(max_alerts_per_hour=6.0).inspect(ledger)

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(f.code == "ALERT_FATIGUE" and f.is_blocking for f in findings)
    report.note(
        "**알람을 줄이는 것은 편의가 아니라 안전이다.** "
        "이 속도면 2주 안에 알람이 꺼지고, 그 다음 사고는 아무도 못 본다."
    )


def test_너무_조용한_것도_문제다() -> None:
    ledger = AlertGate().apply(
        AlertRule(
            alert_labels=("FAULT",),
            dwell=10,
            cooldown_seconds=86_400.0,
        ),
        _signals(200, every=30.0),
    )
    findings = AlertFatiguePolicy(max_suppression_ratio=0.98).inspect(ledger)

    assert any(f.code == "ALERT_TOO_QUIET" for f in findings)
    report.note(
        f"신호 {ledger.raw_signal_count}건 중 {ledger.suppression_ratio:.1%} 를 눌렀다. "
        "**규율이 너무 세면 알람도 조용해지고 사고도 조용해진다.** "
        "시간당 건수로는 이걸 못 잡는다 — 관측 구간이 짧으면 "
        "한 건만 나와도 '하루 14건'으로 환산되기 때문이다."
    )


def test_무엇을_알릴지_정하지_않으면_규칙이_아니다() -> None:
    with pytest.raises(InvariantViolation, match="무엇을 알릴지"):
        AlertRule(alert_labels=())
    with pytest.raises(InvariantViolation, match="1 이상"):
        AlertRule(alert_labels=("FAULT",), dwell=0)


def _reconstruct(device_pipeline):  # noqa: ANN001, ANN202
    """규율 전의 원본 신호를 다시 만든다 — 같은 파이프라인을 규칙만 바꿔 돌린다."""
    from infrastructure.edge.pipeline_runner import DevicePipelineRunner, PipelineSpec

    runner = DevicePipelineRunner(
        PipelineSpec(
            stream_uri=device_pipeline.spec_stream,
            device_id=device_pipeline.device_id,
            contract=device_pipeline.contract,
        )
    )
    rule = AlertRule(
        alert_labels=("FAULT", "OVERLOAD"),
        dwell=1,
        min_confidence=0.0,
        cooldown_seconds=0.0,
        hourly_budget=100_000,
    )
    _, ledger = runner.run(device_pipeline.predict, rule)
    return ledger.alerts
