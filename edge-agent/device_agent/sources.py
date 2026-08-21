"""센서·카메라에서 표본을 받는다. 파이프라인의 첫 단계다. (실습 5-12 ACQUIRE)

여기가 현장마다 가장 많이 달라지는 곳이다. 그래서 Protocol 하나로 고정하고,
구현을 갈아끼운다.

    CsvReplaySource      기록해 둔 CSV 를 **현장 속도로** 흘린다 — 실습·시연·회귀
    SerialSensorSource   실제 계측기 (pyserial)
    CameraSource         실제 카메라 (opencv)

뒤의 둘은 **라이브러리가 없으면 조용히 넘어가지 않고 즉시 실패한다.**
"카메라가 없어서 0으로 채웠습니다"가 현장에서 가장 위험한 코드다.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Sample:
    """표본 하나. 시각과 값들뿐이다."""

    at_seconds: float
    values: tuple[float, ...]
    segment: str = ""
    """이 표본이 속한 구간(배치·로트). 창이 이 경계를 넘으면 안 된다 (실습 5-12)."""

    truth: str = ""
    """현장에서 나중에 붙는 정답. 대개 비어 있다 (O06 — 약 12%)."""


@runtime_checkable
class SampleSource(Protocol):
    """표본을 흘려 준다."""

    def stream(self) -> Iterator[Sample]: ...

    def close(self) -> None: ...


class AcquisitionFailed(RuntimeError):
    """표본을 받을 수 없다. **0으로 채우지 않는다.**"""


class CsvReplaySource:
    """기록해 둔 CSV 를 흘린다.

    `speedup` 으로 시간을 압축한다. 1.0 이면 실제 속도, 0 이면 기다리지 않는다.
    회귀 테스트에서는 0 을 쓴다 — **4일치를 4일 동안 돌릴 수는 없다.**
    """

    def __init__(
        self,
        path: Path,
        *,
        feature_fields: tuple[str, ...],
        device_id: str = "",
        device_field: str = "device_id",
        time_field: str = "timestamp",
        segment_field: str = "batch_id",
        truth_field: str = "condition",
        sample_interval_seconds: float = 10.0,
        speedup: float = 0.0,
        limit: int = 0,
    ) -> None:
        self._path = Path(path)
        self._fields = feature_fields
        self._device_id = device_id
        self._device_field = device_field
        self._time_field = time_field
        self._segment_field = segment_field
        self._truth_field = truth_field
        self._interval = sample_interval_seconds
        self._speedup = speedup
        self._limit = limit
        self._handle = None
        self.skipped_unreadable = 0

    def stream(self) -> Iterator[Sample]:
        if not self._path.is_file():
            raise AcquisitionFailed(f"입력 파일이 없다: {self._path}")

        self._handle = self._path.open("r", encoding="utf-8", newline="")
        reader = csv.DictReader(self._handle)
        missing = [f for f in self._fields if f not in (reader.fieldnames or ())]
        if missing:
            raise AcquisitionFailed(
                f"입력 열이 없다: {missing}. "
                "**계약과 다른 파일이다** — 여기서 멈추는 것이 옳다."
            )

        index = 0
        for row in reader:
            if self._device_id and row.get(self._device_field) != self._device_id:
                continue
            values = self._values(row)
            if values is None:
                # 값이 빠진 표본. **세어 두고 넘어간다** — 조용히 0으로 채우지 않는다.
                self.skipped_unreadable += 1
                continue

            yield Sample(
                at_seconds=index * self._interval,
                values=values,
                segment=str(row.get(self._segment_field, "")),
                truth=str(row.get(self._truth_field, "")),
            )
            index += 1
            if self._limit and index >= self._limit:
                break
            if self._speedup > 0:
                time.sleep(self._interval / self._speedup)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _values(self, row: dict[str, str]) -> tuple[float, ...] | None:
        parsed: list[float] = []
        for name in self._fields:
            raw = (row.get(name) or "").strip()
            if not raw:
                return None
            try:
                parsed.append(float(raw))
            except ValueError:
                return None
        return tuple(parsed)


class SerialSensorSource:
    """실제 계측기에서 읽는다.

    현장 배선은 저마다 다르다. 여기 있는 것은 **가장 흔한 모양** —
    한 줄에 쉼표로 구분된 값이 오는 경우 — 하나뿐이다.
    Modbus·CAN 은 이 자리를 같은 Protocol 로 갈아끼운다.
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115_200,
        field_count: int,
        sample_interval_seconds: float = 10.0,
        timeout: float = 5.0,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._field_count = field_count
        self._interval = sample_interval_seconds
        self._timeout = timeout
        self._serial = None
        self.skipped_unreadable = 0

    def stream(self) -> Iterator[Sample]:
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AcquisitionFailed(
                "pyserial 이 없다. 실제 센서를 읽으려면 설치해야 한다 — "
                "**없다고 0을 흘려보내지 않는다.**"
            ) from exc

        self._serial = serial.Serial(
            self._port, self._baudrate, timeout=self._timeout
        )
        index = 0
        while True:
            line = self._serial.readline().decode("utf-8", errors="replace").strip()
            if not line:
                # 타임아웃. **센서가 죽었을 수도 있다** — 세어 두고 계속한다.
                self.skipped_unreadable += 1
                continue
            parts = line.split(",")
            if len(parts) != self._field_count:
                self.skipped_unreadable += 1
                continue
            try:
                values = tuple(float(p) for p in parts)
            except ValueError:
                self.skipped_unreadable += 1
                continue
            yield Sample(at_seconds=index * self._interval, values=values)
            index += 1

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


class CameraSource:
    """카메라에서 한 장씩 받는다. (캡스톤 주제 1·2)

    이미지 파이프라인은 값이 아니라 **배열**을 흘린다.
    그래서 `Sample.values` 대신 `frames()` 를 쓴다 — 다른 Protocol 이다.
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        width: int,
        height: int,
        interval_seconds: float = 1.0,
    ) -> None:
        self._index = device_index
        self._width = width
        self._height = height
        self._interval = interval_seconds
        self._capture = None
        self.skipped_unreadable = 0

    def frames(self):  # noqa: ANN201
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AcquisitionFailed(
                "opencv 가 없다. 카메라를 읽으려면 설치해야 한다 — "
                "**검은 화면을 정상이라고 판단하게 두지 않는다.**"
            ) from exc

        self._capture = cv2.VideoCapture(self._index)
        if not self._capture.isOpened():
            raise AcquisitionFailed(f"카메라를 열 수 없다: index={self._index}")

        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                # **가장 흔한 현장 사고다.** 렌즈가 가려지거나 케이블이 빠진다.
                self.skipped_unreadable += 1
                time.sleep(self._interval)
                continue
            yield cv2.resize(frame, (self._width, self._height))
            time.sleep(self._interval)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
