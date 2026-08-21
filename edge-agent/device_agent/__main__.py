"""디바이스 에이전트 진입점.

    python -m device_agent run     --slots /var/lib/ondevice/slots --source ...
    python -m device_agent status  --slots /var/lib/ondevice/slots
    python -m device_agent install --slots ... --bundle /tmp/incoming
    python -m device_agent rollback --slots ...

systemd 로 띄운다면 `run` 을 서비스로 두고, 나머지는 손으로 부른다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from device_agent.agent import AgentSettings, DeviceAgent
from device_agent.bundle import BundleRejected
from device_agent.slots import NoPreviousVersion, SlotStore
from device_agent.sources import AcquisitionFailed, CsvReplaySource
from device_agent.store import HttpUplink, NullUplink, Spool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="device_agent")
    parser.add_argument(
        "command", choices=("run", "status", "install", "rollback")
    )
    parser.add_argument("--slots", required=True, type=Path)
    parser.add_argument("--spool", type=Path, default=Path("./spool"))
    parser.add_argument("--state", type=Path, default=Path("./agent-state.json"))
    parser.add_argument("--bundle", type=Path, help="install 할 묶음 디렉터리")
    parser.add_argument("--device-id", default="DEV-01")
    parser.add_argument("--fleet-id", default="line3")
    parser.add_argument("--backend", default="", help="비우면 로컬에만 쌓는다")
    parser.add_argument("--source", type=Path, help="CSV 재생 경로")
    parser.add_argument("--speedup", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args(argv)

    slots = SlotStore(args.slots)

    if args.command == "status":
        print(slots.describe())
        return 0

    if args.command == "install":
        if args.bundle is None:
            parser.error("--bundle 이 필요하다")
        try:
            bundle = slots.activate(args.bundle.name)
        except BundleRejected as exc:
            print(f"거절: {exc}", file=sys.stderr)
            return 2
        print(f"켰다: {bundle.describe()}")
        return 0

    if args.command == "rollback":
        try:
            bundle = slots.rollback()
        except NoPreviousVersion as exc:
            print(str(exc), file=sys.stderr)
            return 3
        print(f"되돌렸다: {bundle.describe()}")
        return 0

    # run
    try:
        bundle = slots.load_active()
    except BundleRejected as exc:
        print(f"올라와 있는 묶음이 없다: {exc}", file=sys.stderr)
        return 2
    if args.source is None:
        parser.error("--source 가 필요하다 (실제 센서는 sources.py 를 갈아끼운다)")

    source = CsvReplaySource(
        args.source,
        feature_fields=bundle.contract.feature_fields,
        device_id=args.device_id,
        sample_interval_seconds=bundle.contract.sample_interval_seconds,
        speedup=args.speedup,
    )
    agent = DeviceAgent(
        AgentSettings(device_id=args.device_id, fleet_id=args.fleet_id),
        bundle,
        source,
        Spool(args.spool),
        HttpUplink(args.backend, args.fleet_id, args.device_id)
        if args.backend
        else NullUplink(),
        slots=slots,
        state_path=args.state,
    )
    agent.install_signal_handlers()

    try:
        run = agent.run(max_samples=args.max_samples)
    except (AcquisitionFailed, BundleRejected) as exc:
        print(f"멈춘다: {exc}", file=sys.stderr)
        return 4

    print(run.render())
    for finding in agent.findings(run):
        print(f"  - {finding.describe()}")
    if agent.is_slower_than_expected():
        print(
            f"  - 느려졌다: p95 {agent.latency_p95():.2f}ms "
            f"(기대 {bundle.expected_p95_ms:.2f}ms) — 실습 5-5"
        )
    print(f"  보내지 못한 기록 {agent.spool.pending_count():,}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
