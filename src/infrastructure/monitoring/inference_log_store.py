"""추론 로그 저장소 — 인메모리 구현. (실습 5-3)

현장에서는 여기가 DynamoDB 나 S3 나 CloudWatch Logs 다.
모듈 6 에서 그쪽으로 옮긴다. **Domain 은 그 사실을 모른다.**

이 구현이 하는 일은 두 가지뿐이다.
    1. 받아 적는다
    2. 시각 구간과 디바이스로 다시 꺼내 준다

두 번째가 없으면 실습 5-4 부터 전부 못 한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from domain.operations.identifiers import DeploymentId
from domain.operations.inference_log import InferenceRecord
from domain.operations.window import ObservationWindow


class InMemoryInferenceLogStore:
    """domain.operations.ports.InferenceLogStore 구현."""

    def __init__(self) -> None:
        self._records: dict[str, list[InferenceRecord]] = {}
        self._current: str = ""

    # -- 쓰기 --------------------------------------------------------------
    def bind(self, deployment_id: DeploymentId) -> None:
        """다음 append 가 어느 배포의 것인지 정한다.

        현장에서는 디바이스가 배포 ID 를 함께 올린다.
        여기서는 실습을 단순하게 하려고 바인딩으로 대신한다.
        """
        self._current = str(deployment_id)
        self._records.setdefault(self._current, [])

    def append(self, records: Sequence[InferenceRecord]) -> int:
        if not self._current:
            raise RuntimeError(
                "어느 배포의 로그인지 정해지지 않았다. bind() 를 먼저 부른다."
            )
        bucket = self._records.setdefault(self._current, [])
        bucket.extend(records)
        bucket.sort(key=lambda r: (r.occurred_at, r.device_id))
        return len(records)

    def attach_ground_truth(
        self, deployment_id: DeploymentId, digest: str, label: str
    ) -> bool:
        """나중에 사람이 정답을 붙인다. (실습 5-11)

        InferenceRecord 는 frozen 이므로 새로 만들어 갈아 끼운다 —
        기록을 제자리에서 고치지 않는다는 뜻이기도 하다.
        """
        from dataclasses import replace

        bucket = self._records.get(str(deployment_id), [])
        for index, record in enumerate(bucket):
            if record.input_digest == digest:
                bucket[index] = replace(record, ground_truth=label)
                return True
        return False

    # -- 읽기 --------------------------------------------------------------
    def records_in(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> Sequence[InferenceRecord]:
        bucket = self._records.get(str(deployment_id), [])
        return tuple(
            record
            for record in bucket
            if window.started_at <= record.occurred_at <= window.ended_at
            and (window.device_id is None or record.device_id == window.device_id)
        )

    def windows_of(
        self, deployment_id: DeploymentId
    ) -> Sequence[ObservationWindow]:
        """이 배포에 대해 만들어 둔 창 목록.

        기본은 '전체를 한 창으로'다. 시간으로 쪼개는 것은 `slice_windows` 가 한다.
        """
        bucket = self._records.get(str(deployment_id), [])
        if not bucket:
            return ()
        return (
            ObservationWindow(
                label="전체",
                started_at=bucket[0].occurred_at,
                ended_at=bucket[-1].occurred_at,
                sample_count=len(bucket),
            ),
        )

    def count(self, deployment_id: DeploymentId) -> int:
        return len(self._records.get(str(deployment_id), []))

    def all_records(self, deployment_id: DeploymentId) -> Sequence[InferenceRecord]:
        return tuple(self._records.get(str(deployment_id), []))

    def clear(self) -> None:
        self._records.clear()
        self._current = ""


def slice_windows(
    records: Sequence[InferenceRecord],
    *,
    hours: int,
    device_id: str | None = None,
    label_prefix: str = "",
) -> list[ObservationWindow]:
    """로그를 시간 간격으로 쪼개 창 목록을 만든다. (실습 5-4)

    창 길이는 현장이 정한다. 짧으면 튀는 것에 놀라고, 길면 늦게 안다.
    """
    if not records:
        return []

    selected = [
        r for r in records if device_id is None or r.device_id == device_id
    ]
    if not selected:
        return []

    first = _parse(selected[0].occurred_at)
    last = _parse(selected[-1].occurred_at)
    span = timedelta(hours=hours)

    windows: list[ObservationWindow] = []
    start = first.replace(minute=0, second=0, microsecond=0)
    index = 0
    while start <= last:
        end = start + span - timedelta(seconds=1)
        count = sum(1 for r in selected if start <= _parse(r.occurred_at) <= end)
        windows.append(
            ObservationWindow(
                label=f"{label_prefix}{start:%m-%d %H시}",
                started_at=start.isoformat(sep=" "),
                ended_at=end.isoformat(sep=" "),
                sample_count=count,
                device_id=device_id,
            )
        )
        start += span
        index += 1
    return windows


def _parse(moment: str) -> datetime:
    return datetime.fromisoformat(moment)
