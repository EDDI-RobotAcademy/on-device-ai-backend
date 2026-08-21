"""로컬 버퍼와 업링크. 파이프라인의 다섯 번째 단계다. (실습 5-12 EMIT, 6-1)

디바이스는 네트워크가 끊긴다. **그때 판단을 버리면 안 된다.**
그래서 먼저 디스크에 쓰고, 회선이 살아나면 보낸다 (store-and-forward).

여기서 지키는 것 셋:

    디스크 상한   무한히 쌓지 않는다. 넘으면 **오래된 것부터** 버리고 **버린 사실을 센다.**
    개인정보      올려서는 안 되는 열은 **디바이스에서** 거른다 (실습 6-1).
                  서버까지 갔다는 건 이미 회선을 건넜다는 뜻이다.
    체크섬        좁은 회선에서 잘려 도착한 묶음을 서버가 알아챌 수 있게 붙인다.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

FORBIDDEN_FIELDS = frozenset(
    {"operator_name", "employee_id", "badge_id", "phone", "email"}
)


@dataclass(slots=True)
class SpoolStats:
    written: int = 0
    dropped_over_capacity: int = 0
    forwarded: int = 0
    forward_failures: int = 0
    stripped_forbidden: int = 0


@dataclass(slots=True)
class Spool:
    """보내지 못한 기록을 담아 두는 곳."""

    root: Path
    max_bytes: int = 32 * 1024 * 1024
    batch_size: int = 360
    """한 묶음에 몇 건. **한 건씩 올리면 요청 수가 곧 비용이고 연결이 곧 전력이다** (실습 6-1)."""

    stats: SpoolStats = field(default_factory=SpoolStats)
    _current: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._current = self.root / "pending.jsonl"

    def append(self, record: dict) -> None:
        clean, stripped = _strip_forbidden(record)
        self.stats.stripped_forbidden += stripped

        line = json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n"
        with self._current.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self.stats.written += 1
        self._enforce_capacity()

    def pending_count(self) -> int:
        if not self._current.is_file():
            return 0
        with self._current.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def take_batch(self) -> tuple[list[dict], str]:
        """앞에서부터 한 묶음을 꺼낸다. **아직 지우지 않는다.**

        보내는 데 실패하면 그대로 남아 있어야 한다.
        """
        if not self._current.is_file():
            return [], ""
        records: list[dict] = []
        with self._current.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= self.batch_size:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
        return records, hashlib.sha256(payload).hexdigest()

    def commit(self, count: int) -> None:
        """보내진 만큼만 앞에서 지운다."""
        if count <= 0 or not self._current.is_file():
            return
        remaining = self._current.read_text(encoding="utf-8").splitlines()[count:]
        temporary = self._current.with_suffix(".tmp")
        temporary.write_text(
            "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8"
        )
        os.replace(temporary, self._current)  # 원자적 교체 — 중간에 죽어도 안 깨진다
        self.stats.forwarded += count

    def _enforce_capacity(self) -> None:
        if not self._current.is_file():
            return
        if self._current.stat().st_size <= self.max_bytes:
            return
        lines = self._current.read_text(encoding="utf-8").splitlines()
        keep = lines[len(lines) // 2 :]  # 오래된 절반을 버린다
        self.stats.dropped_over_capacity += len(lines) - len(keep)
        temporary = self._current.with_suffix(".tmp")
        temporary.write_text("\n".join(keep) + "\n", encoding="utf-8")
        os.replace(temporary, self._current)


def _strip_forbidden(record: dict) -> tuple[dict, int]:
    """올려서는 안 되는 열을 **디바이스에서** 지운다.

    한 번 올라간 것은 지워도 지워지지 않는다 — 백업·복제·로그·캐시에 남는다.
    """
    removed = 0
    clean = {}
    for key, value in record.items():
        if key in FORBIDDEN_FIELDS:
            removed += 1
            continue
        clean[key] = value
    return clean, removed


class HttpUplink:
    """백엔드로 묶음을 올린다. (POST /fleets/{id}/uplinks)

    실패해도 예외를 밖으로 던지지 않는다 — **회선이 끊긴 것은 사고가 아니다.**
    False 를 돌려주고, Spool 이 그대로 들고 있는다.
    """

    def __init__(
        self,
        base_url: str,
        fleet_id: str,
        device_id: str,
        *,
        timeout: float = 10.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._fleet_id = fleet_id
        self._device_id = device_id
        self._timeout = timeout

    def send(
        self,
        records: Sequence[dict],
        checksum: str,
        *,
        window_start: str,
        window_end: str,
        kind: str = "INFERENCE_LOG",
    ) -> bool:
        import base64
        import urllib.error
        import urllib.request

        body = json.dumps(list(records), ensure_ascii=False).encode("utf-8")
        payload = json.dumps(
            {
                "batch": {
                    "device_id": self._device_id,
                    "kind": kind,
                    "window_start": window_start,
                    "window_end": window_end,
                    "record_count": len(records),
                    "payload_bytes": len(body),
                    "checksum": checksum,
                    "fields": sorted(records[0]) if records else [],
                },
                "body_base64": base64.b64encode(body).decode("ascii"),
            }
        ).encode("utf-8")

        request = urllib.request.Request(  # noqa: S310 - 주소는 설정에서 온다
            f"{self._base}/fleets/{self._fleet_id}/uplinks",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, TimeoutError):
            return False


class NullUplink:
    """회선이 없는 자리. 로컬에만 쌓는다.

    현장 시운전에서 실제로 쓴다 — 네트워크 공사가 끝나기 전에 라인이 먼저 돈다.
    """

    def send(self, records, checksum, **kwargs) -> bool:  # noqa: ANN001, ANN003
        return False
