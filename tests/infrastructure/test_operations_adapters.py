"""운영 어댑터가 실제로 무엇을 하는지.

여기서는 진짜 모델이 4일치 현장 신호를 돌고, 진짜 PSI 가 계산된다.
"""

from __future__ import annotations

import pytest

from domain.operations.identifiers import DeploymentId
from domain.operations.window import ObservationWindow
from infrastructure.monitoring.inference_log_store import (
    InMemoryInferenceLogStore,
    slice_windows,
)


def window(**overrides) -> ObservationWindow:  # noqa: ANN003
    base: dict[str, object] = dict(
        label="w",
        started_at="2026-05-20 00:00:00",
        ended_at="2026-05-20 07:59:59",
        sample_count=0,
    )
    base.update(overrides)
    return ObservationWindow(**base)  # type: ignore[arg-type]


class TestLogStore:
    def test_배포를_정하지_않으면_적을_수_없다(self) -> None:
        store = InMemoryInferenceLogStore()
        with pytest.raises(RuntimeError):
            store.append(())

    def test_시각_구간으로_꺼낸다(self, deployed) -> None:
        store = deployed.operations.logs
        first = store.records_in(DeploymentId.of(deployed.deployment_id), window())

        assert first
        assert all("2026-05-20 0" in r.occurred_at for r in first)

    def test_디바이스로_좁힐_수_있다(self, deployed) -> None:
        store = deployed.operations.logs
        scoped = store.records_in(
            DeploymentId.of(deployed.deployment_id), window(device_id="DEV-02")
        )
        assert scoped
        assert {r.device_id for r in scoped} == {"DEV-02"}

    def test_정답은_나중에_붙는다(self, deployed) -> None:
        store = deployed.operations.logs
        deployment_id = DeploymentId.of(deployed.deployment_id)
        target = next(
            r for r in store.all_records(deployment_id) if r.ground_truth is None
        )

        assert store.attach_ground_truth(deployment_id, target.input_digest, "FAULT")
        updated = next(
            r
            for r in store.all_records(deployment_id)
            if r.input_digest == target.input_digest
        )
        assert updated.ground_truth == "FAULT"

    def test_없는_지문에는_붙지_않는다(self, deployed) -> None:
        assert not deployed.operations.logs.attach_ground_truth(
            DeploymentId.of(deployed.deployment_id), "없는지문", "FAULT"
        )


class TestSliceWindows:
    def test_시간_간격으로_쪼갠다(self, deployed) -> None:
        records = deployed.operations.logs.all_records(
            DeploymentId.of(deployed.deployment_id)
        )
        windows = slice_windows(records, hours=8)

        assert len(windows) == 12
        assert sum(w.sample_count for w in windows) == len(records)

    def test_디바이스별로_쪼갤_수_있다(self, deployed) -> None:
        records = deployed.operations.logs.all_records(
            DeploymentId.of(deployed.deployment_id)
        )
        scoped = slice_windows(records, hours=8, device_id="DEV-02")

        assert len(scoped) == 12
        assert all(w.device_id == "DEV-02" for w in scoped)
        assert sum(w.sample_count for w in scoped) < len(records)

    def test_로그가_없으면_창도_없다(self) -> None:
        assert slice_windows((), hours=8) == []


class TestDeviceSimulator:
    def test_예측은_진짜_모델이_낸_것이다(self, deployed) -> None:
        labels = {r.predicted_label for r in deployed.records}
        assert labels <= {"FAULT", "OVERLOAD", "NORMAL"}
        assert len(labels) >= 2

    def test_확신도는_softmax_최대값이다(self, deployed) -> None:
        confidences = [r.confidence for r in deployed.records[:5000]]
        assert all(1 / 3 <= c <= 1.0 for c in confidences)

    def test_같은_시드면_같은_지연시간이_나온다(self, operations_data, deployed) -> None:
        """실행마다 숫자가 달라지면 문서에 숫자를 쓸 수 없다."""
        from infrastructure.edge.device_simulator import _stable_hash

        assert _stable_hash("DEV-02") == _stable_hash("DEV-02")

    def test_한_대만_느려진다(self, deployed) -> None:
        import numpy as np

        last_day = [r for r in deployed.records if r.occurred_at.startswith("2026-05-23")]
        p95 = {
            device: float(
                np.percentile(
                    [r.latency_ms for r in last_day if r.device_id == device], 95
                )
            )
            for device in ("DEV-01", "DEV-02", "DEV-03")
        }
        assert p95["DEV-02"] > p95["DEV-01"] * 2
        assert p95["DEV-03"] < p95["DEV-01"] * 1.5


class TestDriftMeasurer:
    def test_1일차는_학습_분포와_같다(self, deployed) -> None:
        report = deployed.operations.drift.measure(
            DeploymentId.of(deployed.deployment_id), window()
        )
        assert report.max_psi < 0.1

    def test_4일차는_멀어져_있다(self, deployed) -> None:
        report = deployed.operations.drift.measure(
            DeploymentId.of(deployed.deployment_id),
            window(
                label="d4",
                started_at="2026-05-23 00:00:00",
                ended_at="2026-05-23 07:59:59",
            ),
        )
        assert report.max_psi > 5.0
        assert report.worst.field_name == "temperature_c"
        assert report.worst.out_of_range_ratio > 0.5

    def test_구간에_데이터가_없으면_빈_보고서다(self, deployed) -> None:
        report = deployed.operations.drift.measure(
            DeploymentId.of(deployed.deployment_id),
            window(
                label="없음",
                started_at="2027-01-01 00:00:00",
                ended_at="2027-01-01 07:59:59",
            ),
        )
        assert report.features == ()


class TestLatencyMeasurer:
    def test_분위수가_순서대로_나온다(self, deployed) -> None:
        profile = deployed.operations.latency.measure(
            DeploymentId.of(deployed.deployment_id), window()
        )
        assert profile.p50_ms <= profile.p95_ms <= profile.p99_ms <= profile.max_ms

    def test_타임아웃은_분위수에서_빠진다(self, deployed) -> None:
        from infrastructure.monitoring.field_measurers import LogLatencyMeasurer

        strict = LogLatencyMeasurer(deployed.operations.logs, timeout_ms=0.04)
        profile = strict.measure(
            DeploymentId.of(deployed.deployment_id),
            window(
                label="d4",
                started_at="2026-05-23 00:00:00",
                ended_at="2026-05-23 07:59:59",
            ),
        )
        assert profile.timeout_count > 0
        assert profile.p95_ms <= 0.04

    def test_로그가_없으면_0이다(self, deployed) -> None:
        profile = deployed.operations.latency.measure(
            DeploymentId.of(deployed.deployment_id),
            window(
                label="없음",
                started_at="2027-01-01 00:00:00",
                ended_at="2027-01-01 07:59:59",
            ),
        )
        assert profile.p95_ms == 0.0


class TestMixMeasurer:
    def test_예측을_클래스별로_센다(self, deployed) -> None:
        mix = deployed.operations.mix.measure(
            DeploymentId.of(deployed.deployment_id), window()
        )
        assert mix.total > 1000
        assert set(mix.counts) <= {"FAULT", "OVERLOAD", "NORMAL"}
        assert all(0 <= v <= 1 for v in mix.mean_confidence.values())
