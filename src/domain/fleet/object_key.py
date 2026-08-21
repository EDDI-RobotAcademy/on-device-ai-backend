"""Edge의 데이터를 S3에 모아라. (실습 6-2)

**이 파일에 S3 라는 단어가 없다.** 객체 저장소가 무엇이든 같은 문제가 있기 때문이다.

    "어디에 두느냐가 나중에 무엇을 꺼낼 수 있는지를 정한다."

키를 이렇게 짜면:

    uplinks/2026-05-23T09-14-22-DEV-02.json

"지난주 DEV-02 것만 줘"를 하려면 **전부 훑어야 한다.**
객체가 100만 개면 100만 번 훑는다.

이렇게 짜면:

    kind=inference_log/device=DEV-02/date=2026-05-23/hour=09/part-0001.json

접두어로 바로 좁혀진다. 이것이 파티셔닝이고,
Athena·Glue·Spark 가 읽는 방식도 이 규칙이다.

그리고 또 하나. **작은 파일 수만 개는 큰 파일 하나보다 훨씬 비싸다.**
요청 수가 비용이고, 목록 조회가 느려지고, 쿼리 엔진이 매번 파일을 연다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity

RESERVED = ("//", "..", " ")


@dataclass(frozen=True, slots=True)
class ObjectKey:
    """객체 하나의 자리.

    파티션은 순서가 있는 `이름=값` 쌍이다. 순서가 곧 좁혀지는 순서다.
    """

    prefix: str
    partitions: tuple[tuple[str, str], ...]
    filename: str

    def __post_init__(self) -> None:
        if not self.filename.strip():
            raise InvariantViolation("파일 이름이 없다.", subject="filename")
        for name, value in self.partitions:
            if not name.strip() or not str(value).strip():
                raise InvariantViolation(
                    "빈 파티션은 나중에 찾을 수 없는 자리를 만든다.",
                    subject=name or "partition",
                )
        rendered = self.render()
        for token in RESERVED:
            if token in rendered:
                raise InvariantViolation(
                    f"키에 '{token}' 이 들어 있다. 객체 저장소마다 다르게 해석한다.",
                    subject="key",
                )

    def render(self) -> str:
        parts = [self.prefix.strip("/")] if self.prefix else []
        parts += [f"{name}={value}" for name, value in self.partitions]
        parts.append(self.filename)
        return "/".join(parts)

    def partition_prefix(self, depth: int) -> str:
        """앞에서 depth 개까지만 쓴 접두어. 이걸로 좁혀서 조회한다."""
        parts = [self.prefix.strip("/")] if self.prefix else []
        parts += [f"{name}={value}" for name, value in self.partitions[:depth]]
        return "/".join(parts) + "/"

    @property
    def partition_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.partitions)

    def __str__(self) -> str:  # pragma: no cover - 편의
        return self.render()


@dataclass(frozen=True, slots=True)
class KeyLayout:
    """이 저장소가 키를 짜는 규칙.

    규칙이 코드 여기저기 흩어지면 나중에 아무도 못 고친다. 한 군데에 둔다.
    """

    prefix: str = "uplinks"
    order: tuple[str, ...] = ("kind", "device", "date", "hour")
    """좁혀지는 순서. **자주 걸러내는 것부터 앞에 둔다.**"""

    def key_for(
        self,
        *,
        kind: str,
        device_id: str,
        date: str,
        hour: str,
        part: int,
        suffix: str = "jsonl",
    ) -> ObjectKey:
        values = {"kind": kind, "device": device_id, "date": date, "hour": hour}
        return ObjectKey(
            prefix=self.prefix,
            partitions=tuple((name, values[name]) for name in self.order),
            filename=f"part-{part:05d}.{suffix}",
        )

    def prefix_for(self, **filters: str) -> str:
        """주어진 조건으로 좁힐 수 있는 만큼 좁힌 접두어.

        **순서대로 이어져야 좁혀진다.** 중간이 비면 거기서 멈춘다 —
        `device` 만 주고 `kind` 를 안 주면 아무것도 못 좁힌다.
        """
        parts = [self.prefix.strip("/")] if self.prefix else []
        for name in self.order:
            if name not in filters:
                break
            parts.append(f"{name}={filters[name]}")
        return "/".join(parts) + "/"

    def can_narrow(self, **filters: str) -> bool:
        """이 조건으로 전체 스캔을 피할 수 있는가."""
        return bool(self.order) and self.order[0] in filters


@dataclass(frozen=True, slots=True)
class ObjectStats:
    """저장소가 실제로 어떻게 생겼는지 센 것."""

    object_count: int
    total_bytes: int
    distinct_prefixes: int = 0
    smallest_bytes: int = 0
    largest_bytes: int = 0

    @property
    def mean_bytes(self) -> float:
        return self.total_bytes / self.object_count if self.object_count else 0.0

    @property
    def total_mib(self) -> float:
        return self.total_bytes / 1024 / 1024

    def describe(self) -> str:
        return (
            f"객체 {self.object_count:,}개  합계 {self.total_mib:,.1f}MiB  "
            f"평균 {self.mean_bytes / 1024:,.1f}KiB"
        )


@dataclass(frozen=True, slots=True)
class KeyLayoutPolicy:
    """이 키 설계로 나중에 살 수 있는가."""

    min_mean_object_kib: float = 64.0
    """이보다 작은 파일이 수만 개면 조회가 파일 여는 데 시간을 다 쓴다."""

    max_objects_per_prefix: int = 10_000
    """한 접두어 아래 객체가 이만큼 넘으면 목록 조회가 페이지를 넘기기 시작한다."""

    required_partitions: tuple[str, ...] = ("device", "date")
    """이것들로 못 좁히면 '지난주 그 디바이스' 를 못 뽑는다."""

    def inspect(
        self, layout: KeyLayout, stats: ObjectStats
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        missing = [p for p in self.required_partitions if p not in layout.order]
        if missing:
            findings.append(
                Finding(
                    code="LAKE_MISSING_PARTITION",
                    message=(
                        f"파티션에 {missing} 가 없다. "
                        "'지난주 DEV-02 것만' 을 뽑으려면 전부 훑어야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="layout",
                )
            )

        if stats.object_count and stats.mean_bytes / 1024 < self.min_mean_object_kib:
            findings.append(
                Finding(
                    code="LAKE_SMALL_FILES",
                    message=(
                        f"평균 {stats.mean_bytes / 1024:.1f}KiB 짜리 객체가 "
                        f"{stats.object_count:,}개다. "
                        "**작은 파일 수만 개는 큰 파일 하나보다 훨씬 비싸다** — "
                        "요청 수가 비용이고, 쿼리 엔진이 매번 파일을 연다."
                    ),
                    severity=Severity.WARNING,
                    subject="objects",
                    measured=stats.mean_bytes / 1024,
                    threshold=self.min_mean_object_kib,
                )
            )

        if stats.distinct_prefixes:
            per_prefix = stats.object_count / stats.distinct_prefixes
            if per_prefix > self.max_objects_per_prefix:
                findings.append(
                    Finding(
                        code="LAKE_PREFIX_TOO_WIDE",
                        message=(
                            f"접두어 하나에 평균 {per_prefix:,.0f}개가 들어 있다. "
                            "파티션을 더 잘게 나눠야 목록 조회가 끝난다."
                        ),
                        severity=Severity.WARNING,
                        subject="prefix",
                        measured=per_prefix,
                        threshold=float(self.max_objects_per_prefix),
                    )
                )

        if layout.order and layout.order[0] not in ("kind", "date"):
            findings.append(
                Finding(
                    code="LAKE_PARTITION_ORDER",
                    message=(
                        f"'{layout.order[0]}' 이 첫 파티션이다. "
                        "**자주 걸러내는 것부터 앞에 둔다** — 앞이 안 맞으면 뒤는 못 좁힌다."
                    ),
                    severity=Severity.INFO,
                    subject="order",
                )
            )

        return tuple(findings)
