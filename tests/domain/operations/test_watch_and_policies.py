"""HealthWatch Aggregate 와 관측 Policy 들.

전부 손으로 만든 숫자로 돌아간다. 로그도 파일도 필요 없다.
"""

from __future__ import annotations

import pytest

from domain.operations.drift import DriftReport, FeatureDrift
from domain.operations.errors import NoObservationRecorded
from domain.operations.health import (
    HealthMetric,
    HealthReport,
    HealthTimeline,
)
from domain.operations.identifiers import DeploymentId, IncidentId, WatchId
from domain.operations.incident import Incident, IncidentKind, IncidentPolicy
from domain.operations.latency import LatencyProfile
from domain.operations.prediction_mix import PredictionMix
from domain.operations.watch import HealthWatch
from domain.operations.window import ObservationWindow, WindowPolicy
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict

BASELINE = {"NORMAL": 0.78, "OVERLOAD": 0.21, "FAULT": 0.01}


def window(index: int = 0, *, samples: int = 2880) -> ObservationWindow:
    day, hour = 20 + index // 3, (index % 3) * 8
    return ObservationWindow(
        label=f"05-{day:02d} {hour:02d}시",
        started_at=f"2026-05-{day:02d} {hour:02d}:00:00",
        ended_at=f"2026-05-{day:02d} {hour + 7:02d}:59:59",
        sample_count=samples,
    )


def report(index: int = 0, *, psi: float = 0.01, findings=()) -> HealthReport:  # noqa: ANN001
    w = window(index)
    return HealthReport(
        window=w,
        deployment_version=1,
        findings=tuple(findings),
        drift=DriftReport(
            window=w,
            features=(
                FeatureDrift(
                    field_name="temperature_c",
                    psi=psi,
                    mean_shift_sigma=psi,
                    out_of_range_ratio=0.0,
                ),
            ),
        ),
        baseline_mix=BASELINE,
    )


def critical(code: str = "OPS_INPUT_OUT_OF_RANGE") -> Finding:
    return Finding(
        code=code, message="본 적 없는 범위", severity=Severity.CRITICAL, subject="t"
    )


def warning(code: str) -> Finding:
    return Finding(code=code, message="주의", severity=Severity.WARNING, subject="t")


def opened() -> HealthWatch:
    return HealthWatch.open(
        WatchId.of("watch-1"),
        DeploymentId.of("dep-1"),
        baseline_p95_ms=0.0031,
        baseline_mix=dict(BASELINE),
    )


class Test관측기록:
    def test_관측_없이는_최근_결과가_없다(self) -> None:
        with pytest.raises(NoObservationRecorded):
            _ = opened().latest

    def test_창은_시간_순으로만_들어온다(self) -> None:
        watch = opened()
        watch.record(report(3))
        with pytest.raises(InvariantViolation):
            watch.record(report(1))

    def test_같은_창을_다시_재면_덮어쓴다(self) -> None:
        watch = opened()
        watch.record(report(0, psi=0.01))
        watch.record(report(0, psi=5.0))

        assert len(watch.reports) == 1
        assert watch.latest.value_of(HealthMetric.INPUT_PSI) == 5.0

    def test_관측하면_사건이_남는다(self) -> None:
        watch = opened()
        watch.pull_events()
        watch.record(report(0))
        assert [e.event_name for e in watch.pull_events()] == ["ObservationRecorded"]


class Test시간선:
    def timeline(self, values) -> HealthTimeline:  # noqa: ANN001
        return HealthTimeline(
            reports=tuple(report(i, psi=v) for i, v in enumerate(values))
        )

    def test_처음_넘긴_창과_무너진_창은_다르다(self) -> None:
        onset = self.timeline([0.01, 5.0, 0.01, 3.0, 4.0, 4.5]).onset_of(
            HealthMetric.INPUT_PSI, 0.2, consecutive=3
        )
        assert onset.first_exceeded.window_label == "05-20 08시"
        assert onset.sustained_from.window_label == "05-21 00시"
        assert onset.is_sustained
        assert not onset.spike_only

    def test_이어지지_않으면_스파이크다(self) -> None:
        onset = self.timeline([0.01, 5.0, 0.01, 0.01]).onset_of(
            HealthMetric.INPUT_PSI, 0.2, consecutive=3
        )
        assert onset.spike_only
        assert not onset.is_sustained

    def test_한_번도_안_넘기면_아무것도_없다(self) -> None:
        onset = self.timeline([0.01] * 5).onset_of(HealthMetric.INPUT_PSI, 0.2)
        assert onset.first_exceeded is None
        assert not onset.spike_only

    def test_연속_1창이면_처음_넘긴_창이_곧_무너진_창이다(self) -> None:
        onset = self.timeline([0.01, 5.0, 0.01]).onset_of(
            HealthMetric.INPUT_PSI, 0.2, consecutive=1
        )
        assert onset.sustained_from == onset.first_exceeded

    def test_연속_창_수는_1_이상이어야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            self.timeline([0.01]).onset_of(HealthMetric.INPUT_PSI, 0.2, consecutive=0)

    def test_값이_없는_지표는_시간선에_안_찍힌다(self) -> None:
        onset = self.timeline([0.01, 5.0]).onset_of(HealthMetric.LATENCY_P95, 0.05)
        assert onset.points == ()


class Test판정유도:
    def test_판정은_저장하지_않고_소견에서_나온다(self) -> None:
        assert report(0).verdict is Verdict.PASSED
        assert report(0, findings=(warning("W"),)).verdict is Verdict.PASSED_WITH_WARNINGS
        assert report(0, findings=(critical(),)).verdict is Verdict.FAILED


class Test사건:
    def test_아무_일도_없으면_사건이_아니다(self) -> None:
        watch = opened()
        clean = report(0)
        watch.record(clean)
        assert watch.open_incident(IncidentId.of("i-1"), clean, IncidentPolicy()) is None
        assert watch.incidents == ()

    def test_사건에는_근거가_붙는다(self) -> None:
        watch = opened()
        bad = report(0, findings=(critical(),))
        watch.record(bad)
        incident = watch.open_incident(IncidentId.of("i-1"), bad, IncidentPolicy())

        assert incident is not None
        assert incident.kind is IncidentKind.INPUT_DRIFT
        assert incident.findings

    def test_근거_없는_사건은_만들_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            Incident(
                incident_id=IncidentId.of("i-1"),
                kind=IncidentKind.MIXED,
                opened_at="2026-05-23 00:00:00",
                window_label="w",
                deployment_version=1,
                findings=(),
            )

    def test_해결_방법_없이_닫지_않는다(self) -> None:
        watch = opened()
        bad = report(0, findings=(critical(),))
        watch.record(bad)
        watch.open_incident(IncidentId.of("i-1"), bad, IncidentPolicy())

        with pytest.raises(InvariantViolation):
            watch.resolve_incident(IncidentId.of("i-1"), "  ")

        resolved = watch.resolve_incident(IncidentId.of("i-1"), "v1 로 롤백")
        assert not resolved.is_open
        assert watch.open_incidents == ()

    def test_없는_사건은_닫을_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            opened().resolve_incident(IncidentId.of("없음"), "아무거나")

    def test_CRITICAL_하나면_멈춘다(self) -> None:
        should, reason = IncidentPolicy().should_quarantine(
            report(0, findings=(critical(),))
        )
        assert should
        assert "OPS_INPUT_OUT_OF_RANGE" in reason

    def test_경고가_여럿_겹쳐도_멈춘다(self) -> None:
        warnings = tuple(warning(f"W{i}") for i in range(4))
        should, reason = IncidentPolicy(
            quarantine_on_critical=False
        ).should_quarantine(report(0, findings=warnings))
        assert should
        assert "겹쳤다" in reason

    def test_경고_하나로는_멈추지_않는다(self) -> None:
        should, _ = IncidentPolicy().should_quarantine(
            report(0, findings=(warning("W1"),))
        )
        assert not should

    def test_소견_코드로_사건_종류를_가른다(self) -> None:
        policy = IncidentPolicy()
        assert policy.kind_of(report(0, findings=(warning("OPS_LATENCY_JITTER"),))) is (
            IncidentKind.LATENCY
        )
        assert policy.kind_of(report(0, findings=(warning("OPS_LABEL_SURGE"),))) is (
            IncidentKind.PREDICTION_SHIFT
        )
        assert policy.kind_of(
            report(0, findings=(warning("OPS_LATENCY_JITTER"), critical()))
        ) is IncidentKind.MIXED


class Test기준재고정:
    def test_이유_없이_기준을_바꾸지_않는다(self) -> None:
        with pytest.raises(InvariantViolation):
            opened().rebaseline({"NORMAL": 1.0}, "")

    def test_빈_분포는_기준이_될_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            opened().rebaseline({}, "이유는 있다")

    def test_바뀐_기준은_사건으로_남는다(self) -> None:
        watch = opened()
        watch.pull_events()
        previous = watch.rebaseline({"NORMAL": 0.9, "FAULT": 0.1}, "1일차 안정 구간")

        assert previous == BASELINE
        assert watch.baseline_mix == {"NORMAL": 0.9, "FAULT": 0.1}
        assert [e.event_name for e in watch.pull_events()] == ["BaselineReanchored"]


class Test재학습신호:
    def test_근거_없는_요청은_남기지_않는다(self) -> None:
        with pytest.raises(InvariantViolation):
            opened().request_retraining("NOW", ())

    def test_요청은_사건으로_남는다(self) -> None:
        watch = opened()
        watch.pull_events()
        watch.request_retraining("PLAN", ("INPUT_DRIFT",))
        assert [e.event_name for e in watch.pull_events()] == ["RetrainingRequested"]


class Test창:
    def test_표본이_적으면_그_숫자를_믿지_않는다(self) -> None:
        messages = WindowPolicy(min_sample_count=100).inspect(window(0, samples=12))
        assert messages
        assert not WindowPolicy().is_reliable(window(0, samples=12))

    def test_끝이_시작보다_앞설_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            ObservationWindow(
                label="w",
                started_at="2026-05-23 08:00:00",
                ended_at="2026-05-23 00:00:00",
                sample_count=1,
            )

    def test_디바이스를_적으면_그_대만_보는_창이다(self) -> None:
        scoped = ObservationWindow(
            label="w",
            started_at="2026-05-23 00:00:00",
            ended_at="2026-05-23 07:59:59",
            sample_count=1,
            device_id="DEV-02",
        )
        assert scoped.is_device_scoped
        assert "DEV-02" in scoped.describe()


class Test예측분포:
    def test_총변동거리로_이동을_잰다(self) -> None:
        mix = PredictionMix(
            window=window(), counts={"NORMAL": 780, "OVERLOAD": 210, "FAULT": 10}
        )
        assert mix.shift_from(BASELINE) == pytest.approx(0.0, abs=1e-9)

    def test_클래스가_사라진_것을_찾는다(self) -> None:
        mix = PredictionMix(window=window(), counts={"NORMAL": 900, "OVERLOAD": 100})
        assert mix.vanished_from(BASELINE) == ("FAULT",)

    def test_비율이_뛴_클래스를_찾는다(self) -> None:
        mix = PredictionMix(
            window=window(), counts={"NORMAL": 600, "OVERLOAD": 200, "FAULT": 200}
        )
        surged = mix.surged_from(BASELINE, factor=3.0)
        assert surged[0][0] == "FAULT"

    def test_확신도는_개수로_가중한다(self) -> None:
        mix = PredictionMix(
            window=window(),
            counts={"NORMAL": 900, "FAULT": 100},
            mean_confidence={"NORMAL": 0.9, "FAULT": 0.5},
        )
        assert mix.overall_confidence == pytest.approx(0.86)

    def test_음수_개수는_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            PredictionMix(window=window(), counts={"NORMAL": -1})


class Test지연시간:
    def test_분위수가_순서대로여야_한다(self) -> None:
        with pytest.raises(InvariantViolation):
            LatencyProfile(
                window=window(), p50_ms=5.0, p95_ms=1.0, p99_ms=2.0, max_ms=3.0
            )

    def test_기준_대비_배수를_잰다(self) -> None:
        profile = LatencyProfile(
            window=window(), p50_ms=0.02, p95_ms=0.06, p99_ms=0.1, max_ms=0.2
        )
        assert profile.regression_ratio_to(0.003) == pytest.approx(20.0)

    def test_기준이_없으면_배수도_없다(self) -> None:
        profile = LatencyProfile(
            window=window(), p50_ms=0.02, p95_ms=0.06, p99_ms=0.1, max_ms=0.2
        )
        assert profile.regression_ratio_to(0.0) == 0.0
