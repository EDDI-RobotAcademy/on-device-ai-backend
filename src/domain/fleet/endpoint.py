"""실시간 추론 Endpoint를 띄워라. (실습 6-14)

모듈 6 은 지금까지 **모델을 디바이스로 보내는 길**만 봤다.
반대 방향도 있다. 모델을 클라우드에 두고 디바이스가 물어보는 것이다.

    온디바이스   모델이 현장에 있다. 네트워크가 끊겨도 판단한다.
    Endpoint     모델이 클라우드에 있다. 디바이스는 물어본다.

Endpoint 가 유리한 자리가 분명히 있다.

    무거운 모델을 써야 할 때 (디바이스가 못 올린다)
    모델을 자주 바꿔야 할 때 (3,000대에 OTA 하지 않아도 된다)
    라벨링·재검증처럼 **실시간이 아닌** 작업

그러나 이 과정의 주제에서는 대개 답이 아니다. 이유가 셋이다.

    1. 네트워크가 끊기면 **라인이 선다.** 공장 회선은 끊긴다.
    2. 왕복 지연이 사이클 타임을 먹는다. 30ms 사이클에 왕복 40ms 는 못 쓴다.
    3. **요청이 없어도 시간당 과금된다.** 띄워 두고 잊은 엔드포인트가 그렇게 생긴다.

그래서 이 파일은 "띄우는 법"이 아니라 **"띄워도 되는가"**를 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class EndpointVariant:
    """엔드포인트 안의 한 갈래. 가중치로 트래픽을 나눈다 (실습 5-14 의 A/B)."""

    name: str
    model_reference: str
    instance_type: str
    instance_count: int = 1
    weight: float = 1.0
    hourly_cost_usd: float = 0.115

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvariantViolation("갈래 이름이 없다.", subject="name")
        if self.instance_count < 1:
            raise InvariantViolation(
                "인스턴스는 1대 이상이어야 한다.", subject="instance_count"
            )
        if self.weight <= 0:
            raise InvariantViolation("가중치는 0보다 커야 한다.", subject="weight")

    @property
    def monthly_cost_usd(self) -> float:
        """**요청이 하나도 없어도 나가는 돈이다.**"""
        return self.hourly_cost_usd * self.instance_count * 24 * 30


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """무엇을 어떻게 띄울 것인가."""

    name: str
    variants: tuple[EndpointVariant, ...]
    purpose: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvariantViolation("엔드포인트 이름이 없다.", subject="name")
        if not self.variants:
            raise InvariantViolation("갈래가 하나도 없다.", subject="variants")
        if len({v.name for v in self.variants}) != len(self.variants):
            raise InvariantViolation("갈래 이름이 중복된다.", subject="variants")

    @property
    def monthly_cost_usd(self) -> float:
        return sum(v.monthly_cost_usd for v in self.variants)

    @property
    def instance_count(self) -> int:
        return sum(v.instance_count for v in self.variants)

    def describe(self) -> str:
        return (
            f"{self.name} — 갈래 {len(self.variants)}개 / "
            f"인스턴스 {self.instance_count}대 / "
            f"월 ${self.monthly_cost_usd:,.0f}"
        )


@dataclass(frozen=True, slots=True)
class EndpointState:
    """실제로 떠 있는 상태. Infrastructure 가 읽어서 채운다."""

    name: str
    status: str
    variants: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    """(갈래 이름, 현재 가중치)."""

    @property
    def is_serving(self) -> bool:
        return self.status == "InService"


@dataclass(frozen=True, slots=True)
class OnlineInferenceProfile:
    """이 엔드포인트를 현장이 어떻게 쓸 것인가. (실습 6-14)"""

    cycle_time_ms: float
    """현장이 요구하는 응답 시간."""

    network_round_trip_ms: float
    """공장에서 클라우드까지의 왕복 시간. **이건 모델을 줄여도 안 줄어든다.**"""

    inference_ms: float = 5.0
    offline_tolerance_minutes: float = 0.0
    """네트워크가 끊겼을 때 라인이 버틸 수 있는 시간. 0이면 **즉시 선다.**"""

    requests_per_hour: int = 0

    def __post_init__(self) -> None:
        for name in ("cycle_time_ms", "network_round_trip_ms", "inference_ms"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)

    @property
    def total_latency_ms(self) -> float:
        return self.network_round_trip_ms + self.inference_ms


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    """엔드포인트를 띄워도 되는가. (실습 6-14)"""

    max_idle_monthly_usd: float = 200.0
    min_variants_for_ab: int = 2

    def inspect(
        self, spec: EndpointSpec, profile: OnlineInferenceProfile
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if profile.total_latency_ms > profile.cycle_time_ms:
            findings.append(
                Finding(
                    code="EP_CYCLE_TIME_MISSED",
                    message=(
                        f"왕복 {profile.network_round_trip_ms:g}ms + "
                        f"추론 {profile.inference_ms:g}ms = "
                        f"{profile.total_latency_ms:g}ms 인데 "
                        f"사이클 타임은 {profile.cycle_time_ms:g}ms 다. "
                        "**모델을 아무리 줄여도 왕복 시간은 안 줄어든다** — "
                        "이 문제는 경량화로 못 푼다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=spec.name,
                    measured=profile.total_latency_ms,
                    threshold=profile.cycle_time_ms,
                )
            )

        if profile.offline_tolerance_minutes <= 0:
            findings.append(
                Finding(
                    code="EP_NO_OFFLINE_FALLBACK",
                    message=(
                        "네트워크가 끊기면 즉시 판단이 멈춘다. "
                        "**공장 회선은 끊긴다** — 정전, 스위치 교체, 공사. "
                        "온디바이스 모델을 함께 두지 않으면 "
                        "그 시간 동안 라인이 서거나 검사 없이 흘러간다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=spec.name,
                    measured=0.0,
                )
            )

        if spec.monthly_cost_usd > self.max_idle_monthly_usd:
            findings.append(
                Finding(
                    code="EP_ALWAYS_ON_COST",
                    message=(
                        f"월 ${spec.monthly_cost_usd:,.0f} 다. "
                        "**요청이 하나도 없어도 그대로 나간다** — "
                        "엔드포인트는 켜 두는 순간부터 과금된다. "
                        "실습이 끝나면 반드시 지워야 하는 이유다."
                    ),
                    severity=Severity.WARNING,
                    subject=spec.name,
                    measured=spec.monthly_cost_usd,
                    threshold=self.max_idle_monthly_usd,
                )
            )

        if profile.requests_per_hour and spec.instance_count:
            per_instance = profile.requests_per_hour / spec.instance_count
            if per_instance < 60:
                findings.append(
                    Finding(
                        code="EP_UNDERUSED",
                        message=(
                            f"인스턴스 한 대가 시간당 {per_instance:.0f}건을 받는다. "
                            "**분당 한 건짜리 일에 상시 서버를 두고 있다** — "
                            "배치 변환(Batch Transform)이나 서버리스가 맞는 모양이다."
                        ),
                        severity=Severity.WARNING,
                        subject=spec.name,
                        measured=per_instance,
                        threshold=60.0,
                    )
                )

        if len(spec.variants) == 1 and spec.variants[0].instance_count == 1:
            findings.append(
                Finding(
                    code="EP_SINGLE_INSTANCE",
                    message=(
                        "인스턴스가 한 대다. **그 한 대가 재시작되면 전부 멈춘다.** "
                        "현장이 이걸 의존하고 있다면 최소 두 대여야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject=spec.name,
                    measured=1.0,
                    threshold=2.0,
                )
            )

        return tuple(findings)
