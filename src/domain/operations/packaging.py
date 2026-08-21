"""파일로 보낼 것인가, 컨테이너로 보낼 것인가. (실습 5-15)

모듈 6 의 OTA 는 "모델 파일을 내려보낸다"를 전제로 한다.
그런데 현장에는 다른 선택지가 있고, 둘의 성질이 완전히 다르다.

    모델 파일만 보낸다
        9 KiB 를 보낸다. 회선이 좁아도 된다.
        **대신 디바이스에 이미 올바른 런타임이 있어야 한다.**
        전처리 코드도, 라벨 순서도, 후처리도 이미 맞아야 한다.
        모델만 바뀌고 전처리가 그대로면 → 아무 에러 없이 틀린다 (실습 5-12).

    컨테이너로 보낸다
        모델 + 런타임 + 전처리 + 후처리를 한 덩어리로 보낸다.
        **"내 노트북에서는 됐는데"가 사라진다.**
        대신 수십~수백 MiB 다. 그리고 컨테이너 런타임이 도는 보드여야 한다.

경계는 대개 여기다.

    MCU / RTOS          컨테이너가 안 돈다. 파일뿐이다.
    리눅스 게이트웨이   둘 다 된다. 회선과 롤백 요구가 정한다.

정답이 하나가 아니다. 그래서 이 파일은 고르지 않는다. **비교표를 만든다.**
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class PackagingKind(Enum):
    MODEL_FILE = "MODEL_FILE"
    """모델 파일만 내려보낸다."""

    BUNDLE = "BUNDLE"
    """모델 + 전처리 명세 + 라벨을 한 묶음으로. 런타임은 디바이스 것을 쓴다."""

    CONTAINER = "CONTAINER"
    """런타임까지 통째로."""

    @property
    def carries_runtime(self) -> bool:
        return self is PackagingKind.CONTAINER

    @property
    def carries_preprocessing(self) -> bool:
        return self is not PackagingKind.MODEL_FILE


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """디바이스가 무엇을 받을 수 있는가."""

    supports_container: bool
    free_storage_bytes: int
    uplink_bytes_per_second: float
    device_count: int = 1

    def __post_init__(self) -> None:
        if self.free_storage_bytes < 0:
            raise InvariantViolation(
                "저장 공간은 음수일 수 없다.", subject="free_storage_bytes"
            )
        if self.uplink_bytes_per_second <= 0:
            raise InvariantViolation(
                "회선 속도는 0보다 커야 한다.", subject="uplink_bytes_per_second"
            )
        if self.device_count < 1:
            raise InvariantViolation(
                "디바이스 수는 1 이상이어야 한다.", subject="device_count"
            )

    def describe(self) -> str:
        return (
            f"컨테이너 {'가능' if self.supports_container else '불가'} / "
            f"여유 {self.free_storage_bytes / 1024 / 1024:.0f} MiB / "
            f"회선 {self.uplink_bytes_per_second / 1024:.0f} KiB/s / "
            f"{self.device_count:,}대"
        )


@dataclass(frozen=True, slots=True)
class PackagingOption:
    """배포 방식 하나."""

    kind: PackagingKind
    payload_bytes: int
    rollback_seconds: float
    """되돌리는 데 걸리는 시간. **사고가 났을 때 이 숫자가 피해 크기를 정한다.**"""

    def __post_init__(self) -> None:
        if self.payload_bytes <= 0:
            raise InvariantViolation(
                "전송량은 0보다 커야 한다.", subject="payload_bytes"
            )
        if self.rollback_seconds < 0:
            raise InvariantViolation(
                "롤백 시간은 음수일 수 없다.", subject="rollback_seconds"
            )

    def transfer_seconds(self, device: DeviceCapability) -> float:
        return self.payload_bytes / device.uplink_bytes_per_second

    def fleet_bytes(self, device: DeviceCapability) -> int:
        """플릿 전체로 보낼 때의 총 전송량. **여기서 청구서가 나온다** (모듈 6)."""
        return self.payload_bytes * device.device_count

    def describe(self, device: DeviceCapability) -> str:
        return (
            f"{self.kind.value:<12}"
            f"{self.payload_bytes / 1024 / 1024:>9.2f} MiB  "
            f"전송 {self.transfer_seconds(device):>7.1f}s  "
            f"롤백 {self.rollback_seconds:>6.0f}s  "
            f"플릿 총량 {self.fleet_bytes(device) / 1024**3:>7.2f} GiB"
        )


@dataclass(frozen=True, slots=True)
class PackagingComparison:
    """배포 방식 비교표. (실습 5-15)"""

    device: DeviceCapability
    options: tuple[PackagingOption, ...]

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise InvariantViolation(
                "비교하려면 최소 두 가지가 필요하다.", subject="options"
            )

    def option_of(self, kind: PackagingKind) -> PackagingOption | None:
        return next((o for o in self.options if o.kind is kind), None)

    def render(self) -> str:
        lines = [f"[배포 방식] {self.device.describe()}"]
        lines += [f"  {o.describe(self.device)}" for o in self.options]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PackagingPolicy:
    """이 방식으로 이 디바이스에 배포할 수 있는가. (실습 5-15)"""

    max_transfer_seconds: float = 600.0
    max_rollback_seconds: float = 120.0
    storage_headroom: float = 2.0
    """새 것을 받는 동안 **옛 것도 남아 있어야** 롤백이 된다. 그래서 2배가 필요하다."""

    def inspect(
        self, comparison: PackagingComparison, option: PackagingOption
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        device = comparison.device

        if option.kind.carries_runtime and not device.supports_container:
            findings.append(
                Finding(
                    code="PKG_RUNTIME_UNSUPPORTED",
                    message=(
                        "이 디바이스에는 컨테이너 런타임이 없다. "
                        "**MCU/RTOS 에서는 선택지 자체가 아니다** — "
                        "여기서는 파일 배포만 가능하고, "
                        "그래서 전처리 일치를 사람이 지켜야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=option.kind.value,
                )
            )

        needed = option.payload_bytes * self.storage_headroom
        if needed > device.free_storage_bytes:
            findings.append(
                Finding(
                    code="PKG_NO_STORAGE",
                    message=(
                        f"{option.payload_bytes / 1024 / 1024:.0f} MiB 를 받으려면 "
                        f"{needed / 1024 / 1024:.0f} MiB 가 필요하다 "
                        f"(여유 {device.free_storage_bytes / 1024 / 1024:.0f} MiB). "
                        "**옛 것을 지우고 받으면 되돌릴 곳이 없어진다** (실습 6-9)."
                    ),
                    severity=Severity.CRITICAL,
                    subject=option.kind.value,
                    measured=needed,
                    threshold=float(device.free_storage_bytes),
                )
            )

        transfer = option.transfer_seconds(device)
        if transfer > self.max_transfer_seconds:
            findings.append(
                Finding(
                    code="PKG_TRANSFER_TOO_LONG",
                    message=(
                        f"한 대에 {transfer / 60:.0f}분 걸린다. "
                        f"{device.device_count:,}대면 회선이 그동안 잡혀 있다 — "
                        "**업데이트 중에도 라인은 돌고 있다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=option.kind.value,
                    measured=transfer,
                    threshold=self.max_transfer_seconds,
                )
            )

        if option.rollback_seconds > self.max_rollback_seconds:
            findings.append(
                Finding(
                    code="PKG_ROLLBACK_SLOW",
                    message=(
                        f"되돌리는 데 {option.rollback_seconds / 60:.0f}분 걸린다. "
                        "**사고가 난 뒤의 이 시간이 곧 피해 크기다** — "
                        "빠른 롤백은 배포 방식을 고를 때 함께 정해지는 것이지 "
                        "나중에 붙이는 기능이 아니다."
                    ),
                    severity=Severity.WARNING,
                    subject=option.kind.value,
                    measured=option.rollback_seconds,
                    threshold=self.max_rollback_seconds,
                )
            )

        if not option.kind.carries_preprocessing:
            findings.append(
                Finding(
                    code="PKG_PREPROCESSING_NOT_SHIPPED",
                    message=(
                        "모델만 보낸다. 전처리와 라벨 순서는 디바이스에 이미 있는 것을 쓴다. "
                        "**둘이 어긋나도 아무 에러가 안 난다** (실습 5-12) — "
                        "그래서 파일 배포를 고르면 계약 대조를 배포 절차에 넣어야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject=option.kind.value,
                )
            )

        return tuple(findings)
