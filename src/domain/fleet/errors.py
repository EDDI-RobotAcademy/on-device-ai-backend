"""Fleet Context 고유 예외."""

from __future__ import annotations

from domain.shared.errors import (
    DomainException,
    EntityNotFound,
    IllegalStateTransition,
)


class FleetNotFound(EntityNotFound):
    code = "FLEET_NOT_FOUND"


class DeviceNotFound(EntityNotFound):
    code = "DEVICE_NOT_FOUND"


class RolloutNotFound(EntityNotFound):
    code = "ROLLOUT_NOT_FOUND"


class ReleaseNotFound(EntityNotFound):
    code = "RELEASE_NOT_FOUND"


class NotReleasable(IllegalStateTransition):
    """디바이스로 내보낼 수 있는 상태가 아니다."""

    code = "NOT_RELEASABLE"


class RolloutHalted(IllegalStateTransition):
    """실패가 기준을 넘어 스스로 멈췄다."""

    code = "ROLLOUT_HALTED"


class UplinkRejected(DomainException):
    """올려서는 안 되는 것을 올렸다."""

    code = "UPLINK_REJECTED"
