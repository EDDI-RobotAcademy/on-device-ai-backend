"""현장 디바이스가 OTA 에 응답하는 것을 흉내낸다. (실습 6-8, 6-9)

**알리는 것까지는 진짜다.** 실제 IoT Job 이 만들어진다.
**디바이스가 뭐라고 했다는 것은 합성이다.** 정직하게 밝혀 둔다.

이유는 환경에 있다. 디바이스가 결과를 보고하는 API 는 `iot-jobs-data` 인데,
그것은 **디바이스 쪽에서** 호출하는 것이고 테스트 환경(moto)에는 구현이 없다.
그래서 현장이라면 이렇게 됐을 결과를 만들어 넣는다.

    성공     대부분
    실패     설치나 검증에서 떨어진다 — **모델 문제일 수 있다**
    미도달   꺼져 있거나 회선이 끊겼다 — **모델 문제가 아니다**
    대기     아직 아무 말이 없다 — 실패가 아니다

현장에서는 이 클래스가 통째로 사라지고 `IotJobsOtaGateway` 만 남는다.
Domain 도 Application 도 그 차이를 모른다 — 같은 Port 를 구현하기 때문이다.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from domain.fleet.identifiers import RolloutId
from domain.fleet.release import ReleaseBundle
from domain.fleet.rollout import DeviceOutcome


@dataclass(frozen=True, slots=True)
class FleetResponseProfile:
    """현장이 어떻게 응답하는가.

    이 세 숫자가 실습의 시나리오를 정한다. 합계가 1을 넘으면 안 된다.
    """

    failure_rate: float = 0.0
    offline_rate: float = 0.08
    """꺼져 있거나 회선이 끊긴 비율. **0 인 현장은 없다.**"""

    pending_rate: float = 0.05
    """아직 응답이 안 온 비율."""

    forced: dict[str, DeviceOutcome] = field(default_factory=dict)
    """특정 디바이스의 결과를 못박는다. 실습에서 시나리오를 만들 때 쓴다."""


class SimulatedFleetOtaGateway:
    """domain.fleet.ports.OtaGateway 구현 (알림은 진짜, 응답은 합성).

    같은 디바이스는 **몇 번을 물어봐도 같은 답**을 한다.
    실패했던 디바이스가 다음 조회에서 성공하면 실패율이 거짓말을 한다.
    """

    def __init__(
        self,
        inner,  # noqa: ANN001 - IotJobsOtaGateway 등 실제 게이트웨이
        profile: FleetResponseProfile | None = None,
        *,
        seed: int = 20260524,
    ) -> None:
        self._inner = inner
        self._profile = profile or FleetResponseProfile()
        self._seed = seed

    # -- 조회/설정 ---------------------------------------------------------
    @property
    def profile(self) -> FleetResponseProfile:
        return self._profile

    def with_profile(self, profile: FleetResponseProfile) -> SimulatedFleetOtaGateway:
        return SimulatedFleetOtaGateway(self._inner, profile, seed=self._seed)

    def set_profile(self, profile: FleetResponseProfile) -> None:
        self._profile = profile

    # -- Port --------------------------------------------------------------
    def announce(
        self, rollout_id: RolloutId, bundle: ReleaseBundle, device_ids: Sequence[str]
    ) -> str:
        """**진짜 IoT Job 을 만든다.** 여기는 합성이 아니다."""
        return self._inner.announce(rollout_id, bundle, device_ids)

    def collect(
        self, rollout_id: RolloutId, device_ids: Sequence[str]
    ) -> dict[str, DeviceOutcome]:
        return {
            device_id: self._outcome_of(rollout_id, device_id)
            for device_id in device_ids
        }

    def cancel(self, rollout_id: RolloutId, reason: str) -> None:
        self._inner.cancel(rollout_id, reason)

    # -- 내부 --------------------------------------------------------------
    def _outcome_of(self, rollout_id: RolloutId, device_id: str) -> DeviceOutcome:
        forced = self._profile.forced.get(device_id)
        if forced is not None:
            return forced

        # 디바이스마다 고정된 0~1 값. 같은 롤아웃에서 같은 디바이스는 같은 답을 한다.
        bucket = (
            zlib.crc32(f"{self._seed}|{rollout_id}|{device_id}".encode()) % 10_000
        ) / 10_000

        if bucket < self._profile.failure_rate:
            return DeviceOutcome.FAILED
        if bucket < self._profile.failure_rate + self._profile.offline_rate:
            return DeviceOutcome.UNREACHABLE
        if (
            bucket
            < self._profile.failure_rate
            + self._profile.offline_rate
            + self._profile.pending_rate
        ):
            return DeviceOutcome.PENDING
        return DeviceOutcome.SUCCEEDED
