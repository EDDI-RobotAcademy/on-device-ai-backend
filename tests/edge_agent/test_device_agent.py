"""온디바이스 에이전트 참조 구현 — 실제로 도는지 확인한다.

여기서 확인하는 것은 다섯 가지다.

    1. 판정이 백엔드와 **같은가** (StreamingAlertGate == AlertGate)
    2. 잘려 온 묶음을 **거절하는가**
    3. 회선이 끊겨도 **안 버리는가**
    4. 되돌릴 수 있는가, 되돌릴 곳이 없으면 그렇게 말하는가
    5. 다섯 단계가 실제로 도는가 — **진짜 TFLite 모델로**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.operations.alerting import (
    AlertGate,
    AlertRule,
    Signal,
    StreamingAlertGate,
)
from domain.operations.pipeline import PipelineContract, PipelineStage
from domain.shared.errors import InvariantViolation


# ---------------------------------------------------------------------------
# 1. 판정이 백엔드와 같은가
# ---------------------------------------------------------------------------
RULE = AlertRule(
    alert_labels=("FAULT", "OVERLOAD"),
    dwell=3,
    min_confidence=0.6,
    cooldown_seconds=300.0,
    hourly_budget=12,
)


def _signals() -> list[Signal]:
    """튀는 것, 이어지는 것, 확신 낮은 것이 섞인 4시간치."""
    import zlib

    made: list[Signal] = []
    for index in range(1_440):  # 10초 간격 4시간
        seed = zlib.crc32(str(index).encode())
        if index % 97 < 12:
            label, confidence = "FAULT", 0.55 + (seed % 40) / 100.0
        elif index % 53 < 6:
            label, confidence = "OVERLOAD", 0.70 + (seed % 25) / 100.0
        else:
            label, confidence = "NORMAL", 0.95
        made.append(
            Signal(at_seconds=index * 10.0, label=label, confidence=min(confidence, 1.0))
        )
    return made


class Test같은_판정:
    def test_증분_게이트가_배치_게이트와_같은_답을_낸다(self) -> None:
        signals = _signals()

        batch = AlertGate().apply(RULE, signals)

        streaming = StreamingAlertGate(RULE)
        emitted = [a for s in signals if (a := streaming.offer(s)) is not None]
        streaming.close()
        ledger = streaming.ledger()

        assert ledger.alert_count == batch.alert_count
        assert [a.at_seconds for a in ledger.alerts] == [
            a.at_seconds for a in batch.alerts
        ]
        assert ledger.withheld_low_confidence == batch.withheld_low_confidence
        assert ledger.suppressed_by_dwell == batch.suppressed_by_dwell
        assert ledger.suppressed_by_cooldown == batch.suppressed_by_cooldown
        assert len(emitted) <= ledger.alert_count

    def test_시계가_뒤로_뛰면_거부한다(self) -> None:
        gate = StreamingAlertGate(RULE)
        gate.offer(Signal(at_seconds=100.0, label="NORMAL", confidence=0.9))
        with pytest.raises(InvariantViolation, match="시간이 되돌아갔다"):
            gate.offer(Signal(at_seconds=50.0, label="NORMAL", confidence=0.9))


# ---------------------------------------------------------------------------
# 2. 묶음 검증
# ---------------------------------------------------------------------------
class Test묶음_검증:
    def test_진짜_묶음은_통과한다(self, deployed_bundle: Path) -> None:
        from device_agent.bundle import load_bundle

        bundle = load_bundle(deployed_bundle)
        assert bundle.version == "v1.3.0"
        assert len(bundle.contract.class_labels) == 3
        assert bundle.model_path.stat().st_size > 0

    def test_잘려_온_파일은_거절한다(self, deployed_bundle: Path, tmp_path: Path) -> None:
        from device_agent.bundle import BundleRejected, load_bundle

        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "manifest.json").write_bytes(
            (deployed_bundle / "manifest.json").read_bytes()
        )
        original = (deployed_bundle / "model.tflite").read_bytes()
        (broken / "model.tflite").write_bytes(original[: len(original) // 2])

        with pytest.raises(BundleRejected, match="체크섬이 다르다"):
            load_bundle(broken)

    def test_계약이_없으면_거절한다(self, tmp_path: Path) -> None:
        from device_agent.bundle import BundleRejected, checksum_of, load_bundle

        root = tmp_path / "no-contract"
        root.mkdir()
        (root / "model.tflite").write_bytes(b"not-a-real-model")
        (root / "manifest.json").write_text(
            json.dumps({"checksum": checksum_of(root / "model.tflite")}),
            encoding="utf-8",
        )
        with pytest.raises(BundleRejected, match="계약이 불완전하다"):
            load_bundle(root)

    def test_모델에_없는_라벨을_알람으로_걸면_거절한다(
        self, deployed_bundle: Path, tmp_path: Path
    ) -> None:
        from device_agent.bundle import BundleRejected, load_bundle

        root = tmp_path / "bad-rule"
        root.mkdir()
        (root / "model.tflite").write_bytes(
            (deployed_bundle / "model.tflite").read_bytes()
        )
        manifest = json.loads(
            (deployed_bundle / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["alert_rule"]["alert_labels"] = ["없는라벨"]
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(BundleRejected, match="영원히 안 뜬다"):
            load_bundle(root)


# ---------------------------------------------------------------------------
# 3. 회선이 끊겨도 안 버린다
# ---------------------------------------------------------------------------
class Test보관과_전송:
    def test_보내지_못하면_그대로_남는다(self, tmp_path: Path) -> None:
        from device_agent.store import NullUplink, Spool

        spool = Spool(tmp_path / "spool", batch_size=10)
        for index in range(25):
            spool.append({"at_seconds": index * 10.0, "predicted_label": "NORMAL"})

        records, checksum = spool.take_batch()
        assert len(records) == 10
        assert NullUplink().send(records, checksum) is False
        assert spool.pending_count() == 25  # **하나도 안 버렸다**

    def test_보낸_만큼만_지운다(self, tmp_path: Path) -> None:
        from device_agent.store import Spool

        spool = Spool(tmp_path / "spool", batch_size=10)
        for index in range(25):
            spool.append({"at_seconds": index * 10.0})

        records, _ = spool.take_batch()
        spool.commit(len(records))
        assert spool.pending_count() == 15
        assert spool.stats.forwarded == 10

    def test_개인정보는_디바이스에서_지운다(self, tmp_path: Path) -> None:
        from device_agent.store import Spool

        spool = Spool(tmp_path / "spool")
        spool.append(
            {
                "at_seconds": 0.0,
                "predicted_label": "FAULT",
                "operator_name": "홍길동",
                "badge_id": "A-1029",
            }
        )
        stored = json.loads((tmp_path / "spool" / "pending.jsonl").read_text())
        assert "operator_name" not in stored
        assert "badge_id" not in stored
        assert spool.stats.stripped_forbidden == 2

    def test_상한을_넘으면_오래된_것부터_버리고_센다(self, tmp_path: Path) -> None:
        from device_agent.store import Spool

        spool = Spool(tmp_path / "spool", max_bytes=2_000)
        for index in range(400):
            spool.append({"at_seconds": index * 10.0, "predicted_label": "NORMAL"})

        assert spool.stats.dropped_over_capacity > 0
        assert (tmp_path / "spool" / "pending.jsonl").stat().st_size <= 4_000


# ---------------------------------------------------------------------------
# 4. 슬롯과 롤백
# ---------------------------------------------------------------------------
class Test슬롯:
    def test_받는_동안_옛_것을_안_지운다(
        self, deployed_bundle: Path, tmp_path: Path
    ) -> None:
        from device_agent.bundle import load_bundle, write_bundle
        from device_agent.slots import SlotStore

        slots = SlotStore(tmp_path / "slots")
        current = load_bundle(deployed_bundle)
        _copy_bundle(deployed_bundle, slots.root / "a")
        slots.activate("a")

        staged = slots.stage_slot()
        assert staged.name == "b"
        write_bundle(
            staged,
            version="v1.4.0",
            model_bytes=current.model_path.read_bytes(),
            contract=current.contract,
            alert_rule=current.alert_rule,
        )
        # 아직 표시를 안 옮겼다 — **지금 도는 것은 그대로다.**
        assert slots.load_active().version == "v1.3.0"

        slots.install(staged)
        assert slots.load_active().version == "v1.4.0"

    def test_되돌리면_표시_하나만_옮긴다(
        self, deployed_bundle: Path, tmp_path: Path
    ) -> None:
        from device_agent.bundle import load_bundle, write_bundle
        from device_agent.slots import SlotStore

        slots = SlotStore(tmp_path / "slots")
        current = load_bundle(deployed_bundle)
        _copy_bundle(deployed_bundle, slots.root / "a")
        slots.activate("a")
        write_bundle(
            slots.root / "b",
            version="v1.4.0",
            model_bytes=current.model_path.read_bytes(),
            contract=current.contract,
            alert_rule=current.alert_rule,
        )
        slots.activate("b")

        rolled = slots.rollback()
        assert rolled.version == "v1.3.0"
        assert slots.active_slot() == "a"

    def test_되돌릴_곳이_없으면_그렇게_말한다(
        self, deployed_bundle: Path, tmp_path: Path
    ) -> None:
        from device_agent.slots import NoPreviousVersion, SlotStore

        slots = SlotStore(tmp_path / "slots")
        _copy_bundle(deployed_bundle, slots.root / "a")
        slots.activate("a")

        with pytest.raises(NoPreviousVersion, match="첫 배포였다"):
            slots.rollback()

    def test_깨진_묶음으로는_켜지지_않는다(
        self, deployed_bundle: Path, tmp_path: Path
    ) -> None:
        from device_agent.bundle import BundleRejected
        from device_agent.slots import SlotStore

        slots = SlotStore(tmp_path / "slots")
        _copy_bundle(deployed_bundle, slots.root / "a")
        slots.activate("a")

        broken = slots.root / "b"
        broken.mkdir(exist_ok=True)
        (broken / "model.tflite").write_bytes(b"truncated")
        (broken / "manifest.json").write_text(
            json.dumps({"checksum": "deadbeef"}), encoding="utf-8"
        )
        with pytest.raises(BundleRejected):
            slots.activate("b")
        assert slots.active_slot() == "a"  # **표시는 그대로다**


# ---------------------------------------------------------------------------
# 5. 다섯 단계가 실제로 돈다
# ---------------------------------------------------------------------------
class Test파이프라인:
    @pytest.fixture(scope="class")
    def agent_run(self, deployed_bundle, operations_data, tmp_path_factory):  # noqa: ANN001, ANN201
        from device_agent.agent import AgentSettings, DeviceAgent
        from device_agent.bundle import load_bundle
        from device_agent.sources import CsvReplaySource
        from device_agent.store import NullUplink, Spool

        bundle = load_bundle(deployed_bundle)
        workspace = tmp_path_factory.mktemp("agent")
        source = CsvReplaySource(
            operations_data.stream,
            feature_fields=bundle.contract.feature_fields,
            device_id="DEV-01",
            sample_interval_seconds=bundle.contract.sample_interval_seconds,
            speedup=0.0,
            limit=4_000,
        )
        agent = DeviceAgent(
            AgentSettings(device_id="DEV-01", stride=3, uplink_every=500),
            bundle,
            source,
            Spool(workspace / "spool"),
            NullUplink(),
            state_path=workspace / "state.json",
        )
        run = agent.run()
        return agent, run

    def test_다섯_단계가_순서대로_돈다(self, agent_run) -> None:  # noqa: ANN001
        _, run = agent_run
        assert [s.stage for s in run.stages] == list(PipelineStage)
        assert run.acquired > 0
        assert run.answered > 0

    def test_진짜_TFLite_모델이_진짜_답을_낸다(self, agent_run) -> None:  # noqa: ANN001
        agent, run = agent_run
        infer = run.stage_of(PipelineStage.INFER)
        assert infer.succeeded > 100
        assert agent.latency_p95() > 0.0

    def test_구간_경계를_넘는_창은_판단하지_않는다(self, agent_run) -> None:  # noqa: ANN001
        _, run = agent_run
        preprocess = run.stage_of(PipelineStage.PREPROCESS)
        assert preprocess.dropped > 0
        assert "구간 경계를 넘는 창" in preprocess.reason_counts

    def test_백엔드와_같은_Policy_로_자기를_판정한다(self, agent_run) -> None:  # noqa: ANN001
        agent, run = agent_run
        findings = agent.findings(run)
        # 계약은 묶음에서 그대로 왔으므로 불일치 소견은 없어야 한다
        assert not any(f.code == "PIPE_CONTRACT_MISMATCH" for f in findings)

    def test_회선이_없으면_로컬에_쌓인다(self, agent_run) -> None:  # noqa: ANN001
        agent, _ = agent_run
        assert agent.spool.pending_count() > 0
        assert agent.spool.stats.forward_failures > 0

    def test_상태가_디스크에_남는다(self, agent_run) -> None:  # noqa: ANN001
        agent, run = agent_run
        assert agent.state.acquired == run.acquired
        assert agent.state.alerts == run.emitted_alerts

    def test_계약이_다른_묶음은_소견으로_잡힌다(self, agent_run) -> None:  # noqa: ANN001
        agent, run = agent_run
        other = PipelineContract(
            input_shape=run.contract.input_shape,
            sample_interval_seconds=30.0,  # 학습은 10초였다
            feature_fields=run.contract.feature_fields,
            normalization=dict(run.contract.normalization),
            class_labels=run.contract.class_labels,
        )
        gaps = run.contract.differences_from(other)
        assert any("표본 간격" in gap for gap in gaps)


def _copy_bundle(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "model.tflite"):
        (target / name).write_bytes((source / name).read_bytes())
