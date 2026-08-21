"""디바이스에서 발생한 데이터를 Cloud로 보내라. (실습 6-1)

가장 먼저 정해야 하는 것은 "어떻게 보낼 것인가"가 아니라
**"무엇을 보낼 것인가"** 다.

전부 보내면 세 군데가 터진다.

    대역폭   현장 회선은 대개 좁다. 3,000대가 동시에 올리면 더 좁아진다.
    비용     S3 저장은 싸지만 전송과 요청 수는 싸지 않다.
    프라이버시 원본 신호에는 남기면 안 되는 것이 섞여 있을 수 있다.

그래서 디바이스는 **요약을 항상, 원본은 조금만** 올린다.

    항상   추론 로그 요약 (모듈 5 의 관측에 필요한 것)
    가끔   원본 신호 표본 (모듈 6-4 의 재학습에 필요한 것)
    절대   개인 식별 정보

마지막이 중요하다. **한 번 올라간 것은 지워도 지워지지 않는다** —
백업, 복제, 로그, 캐시에 남는다. 올리지 않는 것이 유일한 방법이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class UplinkKind(Enum):
    INFERENCE_LOG = "INFERENCE_LOG"
    """추론 판단 기록. 작고, 항상 올린다. (모듈 5)"""

    RAW_SAMPLE = "RAW_SAMPLE"
    """원본 신호 표본. 크고, 골라서 올린다. (실습 6-4 의 재료)"""

    HEALTH_REPORT = "HEALTH_REPORT"
    """디바이스 상태. 아주 작고, 주기적으로 올린다."""

    INCIDENT = "INCIDENT"
    """사건. 드물지만 올라오면 그 구간의 원본을 함께 붙인다."""


@dataclass(frozen=True, slots=True)
class UplinkBatch:
    """디바이스가 한 번에 올린 묶음.

    한 건씩 올리지 않는다. 요청 수가 곧 비용이고, 연결 수립이 곧 전력이다.
    """

    device_id: str
    kind: UplinkKind
    window_start: str
    window_end: str
    record_count: int
    payload_bytes: int
    checksum: str = ""
    fields: tuple[str, ...] = field(default_factory=tuple)
    """이 묶음에 실려 있는 열 이름들. 개인정보 검사가 여기를 본다."""

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise InvariantViolation(
                "어느 디바이스가 올린 것인지 없으면 나중에 되짚을 수 없다.",
                subject="device_id",
            )
        if self.record_count < 0 or self.payload_bytes < 0:
            raise InvariantViolation("음수 크기는 없다.", subject="size")
        if self.window_start > self.window_end:
            raise InvariantViolation("끝이 시작보다 앞이다.", subject="window")

    @property
    def bytes_per_record(self) -> float:
        return self.payload_bytes / self.record_count if self.record_count else 0.0

    @property
    def payload_kib(self) -> float:
        return self.payload_bytes / 1024

    def describe(self) -> str:
        return (
            f"{self.device_id} {self.kind.value} "
            f"{self.record_count:,}건 {self.payload_kib:,.1f}KiB "
            f"({self.window_start} ~ {self.window_end})"
        )


@dataclass(frozen=True, slots=True)
class UplinkPolicy:
    """무엇을 올려도 되는가.

    이 숫자들은 **현장 회선과 계약**이 정한다. 개발자가 정하는 것이 아니다.
    """

    max_batch_kib: float = 512.0
    """한 묶음의 상한. 좁은 회선에서 큰 묶음은 타임아웃으로 끝난다."""

    daily_budget_kib_per_device: float = 20_480.0
    """디바이스 한 대의 하루 전송 예산 (기본 20MiB)."""

    forbidden_fields: frozenset[str] = frozenset(
        {"operator_name", "employee_id", "badge_id", "phone", "email"}
    )
    """**절대 올리지 않는 것.** 한 번 올라가면 지워지지 않는다."""

    min_records_per_batch: int = 10
    """이보다 적으면 묶는 의미가 없다. 요청 수가 비용이다."""

    require_checksum: bool = True

    def inspect(
        self, batch: UplinkBatch, *, sent_today_kib: float = 0.0
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        leaked = sorted(set(batch.fields) & self.forbidden_fields)
        for name in leaked:
            findings.append(
                Finding(
                    code="UPLINK_FORBIDDEN_FIELD",
                    message=(
                        f"'{name}' 는 올려서는 안 되는 열이다. "
                        "**한 번 올라간 것은 지워도 지워지지 않는다** — "
                        "백업·복제·로그·캐시에 남는다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=name,
                )
            )

        if batch.payload_kib > self.max_batch_kib:
            findings.append(
                Finding(
                    code="UPLINK_BATCH_TOO_LARGE",
                    message=(
                        "한 묶음이 너무 크다. 좁은 회선에서는 통째로 실패하고, "
                        "재시도하면 같은 크기를 또 보낸다."
                    ),
                    severity=Severity.WARNING,
                    subject=batch.device_id,
                    measured=batch.payload_kib,
                    threshold=self.max_batch_kib,
                )
            )

        if batch.record_count < self.min_records_per_batch:
            findings.append(
                Finding(
                    code="UPLINK_TOO_CHATTY",
                    message=(
                        f"{batch.record_count}건씩 올리고 있다. "
                        "요청 수가 곧 비용이고, 연결 수립이 곧 전력이다."
                    ),
                    severity=Severity.WARNING,
                    subject=batch.device_id,
                    measured=float(batch.record_count),
                    threshold=float(self.min_records_per_batch),
                )
            )

        projected = sent_today_kib + batch.payload_kib
        if projected > self.daily_budget_kib_per_device:
            findings.append(
                Finding(
                    code="UPLINK_OVER_DAILY_BUDGET",
                    message=(
                        "하루 전송 예산을 넘는다. "
                        "3,000대가 같이 넘으면 회선이 아니라 청구서가 먼저 터진다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=batch.device_id,
                    measured=projected,
                    threshold=self.daily_budget_kib_per_device,
                )
            )

        if self.require_checksum and not batch.checksum:
            findings.append(
                Finding(
                    code="UPLINK_NO_CHECKSUM",
                    message=(
                        "체크섬이 없다. 좁은 회선에서 잘려 올라온 파일을 "
                        "온전한 것으로 착각하게 된다."
                    ),
                    severity=Severity.WARNING,
                    subject=batch.device_id,
                )
            )

        return tuple(findings)

    def accepts(self, batch: UplinkBatch, *, sent_today_kib: float = 0.0) -> bool:
        return not any(
            f.severity is Severity.CRITICAL
            for f in self.inspect(batch, sent_today_kib=sent_today_kib)
        )
