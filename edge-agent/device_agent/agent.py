"""다섯 단계를 실제로 도는 루프. (실습 5-12 의 참조 구현)

    입력 수집 → 전처리 → 추론 → 후처리 → 알람/저장

`infrastructure/edge/pipeline_runner.py` 와 무엇이 다른가:

    pipeline_runner   백엔드에서 **한 번에** 돌려 단계별 통계를 낸다 (실습 5-12)
    DeviceAgent       보드에서 **표본이 올 때마다** 돈다. 미래를 모른다.

같은 판정 코드를 쓴다 — `PipelineContract`, `AlertRule`.
다만 알람은 증분판(`StreamingAlertGate`)을 쓴다. 결과는 같다.

에이전트가 지키는 것:

    끝날 때 상태를 남긴다      전원이 나가도 다음 부팅이 이어받는다
    회선이 끊겨도 안 버린다    로컬에 쌓고 살아나면 보낸다
    느려지면 알아챈다          expected_p95_ms 를 넘으면 소견을 남긴다 (실습 5-5)
    되돌릴 수 있게 둔다        슬롯 두 개를 항상 유지한다 (실습 6-9)
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from domain.operations.alerting import Alert, AlertRule, Signal, StreamingAlertGate
from domain.operations.pipeline import (
    PipelineContract,
    PipelinePolicy,
    PipelineRun,
    PipelineStage,
    StageOutcome,
)
from device_agent.bundle import DeployedBundle
from device_agent.preprocess import WindowBuilder
from device_agent.runtime import TfliteRuntime
from device_agent.slots import SlotStore
from device_agent.sources import SampleSource
from device_agent.store import Spool


@dataclass(frozen=True, slots=True)
class AgentSettings:
    device_id: str
    fleet_id: str = "line3"
    stride: int = 1
    uplink_every: int = 360
    """이만큼 쌓이면 한 번 올린다. **한 건씩 올리지 않는다** (실습 6-1)."""

    heartbeat_every: int = 3_600
    epoch: str = "2026-05-23 00:00:00"
    """표본 시각의 기준점. 로그에 절대 시각을 남기기 위한 것이다."""

    max_p95_ratio: float = 1.5
    """기대 p95 의 이 배를 넘으면 느려졌다고 본다 (실습 5-5)."""


@dataclass(slots=True)
class AgentState:
    """부팅 사이를 건너가는 상태. **전원이 나가도 이어받는다.**"""

    acquired: int = 0
    inferred: int = 0
    answered: int = 0
    withheld: int = 0
    alerts: int = 0
    unreadable: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "acquired": self.acquired,
                "inferred": self.inferred,
                "answered": self.answered,
                "withheld": self.withheld,
                "alerts": self.alerts,
                "unreadable": self.unreadable,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> AgentState:
        data = json.loads(raw)
        return cls(**{k: int(v) for k, v in data.items()})


class DeviceAgent:
    """보드 위에서 도는 것."""

    def __init__(
        self,
        settings: AgentSettings,
        bundle: DeployedBundle,
        source: SampleSource,
        spool: Spool,
        uplink,  # noqa: ANN001 - HttpUplink | NullUplink
        *,
        slots: SlotStore | None = None,
        state_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._bundle = bundle
        self._source = source
        self._spool = spool
        self._uplink = uplink
        self._slots = slots
        self._state_path = state_path

        self._windows = WindowBuilder(bundle.contract, stride=settings.stride)
        self._runtime = TfliteRuntime(
            str(bundle.model_path), bundle.contract.class_labels
        )
        self._gate = StreamingAlertGate(bundle.alert_rule)
        self._state = self._restore()
        self._stopping = False
        self._epoch = datetime.strptime(settings.epoch, "%Y-%m-%d %H:%M:%S")

    # -- 수명 --------------------------------------------------------------
    def install_signal_handlers(self) -> None:
        """SIGTERM 을 받으면 **지금 창을 끝내고** 멈춘다.

        중간에 죽으면 마지막 묶음이 안 올라간다. 그건 잃어도 되는 것이 아니다.
        """
        for name in ("SIGTERM", "SIGINT"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), self._request_stop)

    def _request_stop(self, *_args) -> None:  # noqa: ANN002
        self._stopping = True

    def run(self, *, max_samples: int = 0) -> PipelineRun:
        """다섯 단계를 돈다. 돌려주는 것은 실습 5-12 와 **같은 모양**의 보고서다."""
        started = time.perf_counter()
        stage_ms = {stage: 0.0 for stage in PipelineStage}
        alerts: list[Alert] = []

        try:
            for sample in self._source.stream():
                if self._stopping:
                    break
                self._state.acquired += 1

                mark = time.perf_counter()
                window = self._windows.offer(sample)
                stage_ms[PipelineStage.PREPROCESS] += (time.perf_counter() - mark) * 1000
                if window is None:
                    continue

                mark = time.perf_counter()
                prediction = self._runtime.predict(window)
                stage_ms[PipelineStage.INFER] += (time.perf_counter() - mark) * 1000
                self._state.inferred += 1
                self._state.latency_ms.append(prediction.latency_ms)

                mark = time.perf_counter()
                answered = prediction.confidence >= self._bundle.alert_rule.min_confidence
                if not answered:
                    self._state.withheld += 1
                stage_ms[PipelineStage.POSTPROCESS] += (time.perf_counter() - mark) * 1000

                mark = time.perf_counter()
                if answered:
                    self._state.answered += 1
                    alert = self._gate.offer(
                        Signal(
                            at_seconds=sample.at_seconds,
                            label=prediction.label,
                            confidence=prediction.confidence,
                        )
                    )
                    if alert is not None:
                        alerts.append(alert)
                        self._state.alerts += 1
                    self._record(sample, prediction, alert)
                stage_ms[PipelineStage.EMIT] += (time.perf_counter() - mark) * 1000

                if self._state.answered % self._settings.uplink_every == 0:
                    self.flush()
                if max_samples and self._state.acquired >= max_samples:
                    break
        finally:
            self._source.close()
            closing = self._gate.close()
            if closing is not None:
                alerts.append(closing)
            self.flush()
            self._persist()

        stage_ms[PipelineStage.ACQUIRE] = max(
            0.0,
            (time.perf_counter() - started) * 1000 - sum(stage_ms.values()),
        )
        self._state.unreadable = getattr(self._source, "skipped_unreadable", 0)
        return self._report(stage_ms)

    # -- 업링크 ------------------------------------------------------------
    def flush(self) -> bool:
        """쌓인 것을 올린다. **실패해도 안 버린다.**"""
        records, checksum = self._spool.take_batch()
        if not records:
            return True
        ok = self._uplink.send(
            records,
            checksum,
            window_start=self._stamp(records[0]["at_seconds"]),
            window_end=self._stamp(records[-1]["at_seconds"]),
        )
        if ok:
            self._spool.commit(len(records))
        else:
            self._spool.stats.forward_failures += 1
        return ok

    # -- 보고 --------------------------------------------------------------
    def report(self) -> PipelineRun:
        return self._report({stage: 0.0 for stage in PipelineStage})

    def findings(self, run: PipelineRun):  # noqa: ANN201
        """백엔드와 **같은 Policy 로** 자기 상태를 판정한다."""
        return PipelinePolicy().inspect(run, trained_contract=self._bundle.contract)

    def latency_p95(self) -> float:
        if not self._state.latency_ms:
            return 0.0
        return float(np.percentile(np.array(self._state.latency_ms), 95))

    def is_slower_than_expected(self) -> bool:
        """느려졌는가 (실습 5-5). 팬이 죽으면 여기가 먼저 움직인다."""
        expected = self._bundle.expected_p95_ms
        if expected <= 0:
            return False
        return self.latency_p95() > expected * self._settings.max_p95_ratio

    def rollback(self) -> DeployedBundle:
        """직전 버전으로 되돌린다 (실습 6-9). 슬롯이 없으면 예외가 난다."""
        if self._slots is None:
            raise RuntimeError("슬롯 저장소가 없다. 되돌릴 대상을 모른다")
        return self._slots.rollback()

    # -- 내부 --------------------------------------------------------------
    def _record(self, sample, prediction, alert) -> None:  # noqa: ANN001
        self._spool.append(
            {
                "device_id": self._settings.device_id,
                "model_version_id": self._bundle.model_version_id,
                "release_version": self._bundle.version,
                "at_seconds": sample.at_seconds,
                "occurred_at": self._stamp(sample.at_seconds),
                "predicted_label": prediction.label,
                "confidence": round(prediction.confidence, 6),
                "latency_ms": round(prediction.latency_ms, 4),
                "alerted": alert is not None,
                "ground_truth": sample.truth,
            }
        )

    def _stamp(self, at_seconds: float) -> str:
        return (self._epoch + timedelta(seconds=at_seconds)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def _report(self, stage_ms: dict) -> PipelineRun:  # noqa: ANN001
        windows = self._windows
        return PipelineRun(
            device_id=self._settings.device_id,
            contract=self._bundle.contract,
            stages=(
                StageOutcome(
                    stage=PipelineStage.ACQUIRE,
                    attempted=self._state.acquired + self._state.unreadable,
                    succeeded=self._state.acquired,
                    duration_ms=stage_ms[PipelineStage.ACQUIRE],
                    reason_counts=(
                        {"판독 불가": self._state.unreadable}
                        if self._state.unreadable
                        else {}
                    ),
                ),
                StageOutcome(
                    stage=PipelineStage.PREPROCESS,
                    attempted=windows.attempted,
                    succeeded=self._state.inferred,
                    duration_ms=stage_ms[PipelineStage.PREPROCESS],
                    reason_counts=(
                        {"구간 경계를 넘는 창": windows.dropped_segment_boundary}
                        if windows.dropped_segment_boundary
                        else {}
                    ),
                ),
                StageOutcome(
                    stage=PipelineStage.INFER,
                    attempted=self._state.inferred,
                    succeeded=self._state.inferred,
                    duration_ms=stage_ms[PipelineStage.INFER],
                ),
                StageOutcome(
                    stage=PipelineStage.POSTPROCESS,
                    attempted=self._state.inferred,
                    succeeded=self._state.answered,
                    duration_ms=stage_ms[PipelineStage.POSTPROCESS],
                    reason_counts=(
                        {"확신 부족": self._state.withheld}
                        if self._state.withheld
                        else {}
                    ),
                ),
                StageOutcome(
                    stage=PipelineStage.EMIT,
                    attempted=self._state.answered,
                    succeeded=self._state.answered - self._spool.stats.dropped_over_capacity,
                    duration_ms=stage_ms[PipelineStage.EMIT],
                    reason_counts=(
                        {"버퍼 상한 초과": self._spool.stats.dropped_over_capacity}
                        if self._spool.stats.dropped_over_capacity
                        else {}
                    ),
                ),
            ),
            emitted_alerts=self._state.alerts,
            withheld=self._state.withheld,
        )

    def _restore(self) -> AgentState:
        if self._state_path and self._state_path.is_file():
            try:
                return AgentState.from_json(
                    self._state_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # 깨진 상태 파일은 버린다. 세는 것을 못 세는 것보다 낫다.
        return AgentState()

    def _persist(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(self._state.to_json(), encoding="utf-8")
        temporary.replace(self._state_path)  # 원자적 교체

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def bundle(self) -> DeployedBundle:
        return self._bundle

    @property
    def gate(self) -> StreamingAlertGate:
        return self._gate

    @property
    def spool(self) -> Spool:
        return self._spool
