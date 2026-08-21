"""Memory와 Compute 병목을 직접 찾아라. (실습 4-9)

층이 느린 이유는 둘 중 하나다.

    계산이 많다        → COMPUTE_BOUND. 연산을 줄이거나 더 빠른 연산기를 쓴다.
    데이터를 나른다    → MEMORY_BOUND.  더 빨리 계산해도 소용없다. 옮기는 게 느리다.

가르는 기준은 **산술 강도(arithmetic intensity)** — 1바이트를 나를 때마다 몇 번 계산하는가.

    intensity = MAC 수 / 옮긴 바이트 수

그리고 기계마다 균형점이 있다.

    machine_balance = 연산 성능(MAC/s) ÷ 메모리 대역폭(byte/s)

    intensity < machine_balance  → 메모리가 못 따라온다 (MEMORY_BOUND)
    intensity > machine_balance  → 연산기가 못 따라온다 (COMPUTE_BOUND)

이 구분을 모르면 최적화를 엉뚱한 곳에 한다.
메모리 병목인 층을 INT8 로 바꾸면 빨라지고, 연산 병목인 층은 그대로다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class BottleneckKind(Enum):
    MEMORY_BOUND = "MEMORY_BOUND"
    COMPUTE_BOUND = "COMPUTE_BOUND"
    BALANCED = "BALANCED"
    NEGLIGIBLE = "NEGLIGIBLE"
    """연산도 데이터도 거의 없는 층. 병목이 아니다."""


@dataclass(frozen=True, slots=True)
class LayerCost:
    """층 하나가 치르는 비용."""

    name: str
    kind: str
    mac_count: int
    weight_bytes: int
    activation_bytes: int
    """입력 + 출력. 이 층을 지나가며 실제로 옮긴 데이터다."""

    def __post_init__(self) -> None:
        for attribute in ("mac_count", "weight_bytes", "activation_bytes"):
            if getattr(self, attribute) < 0:
                raise InvariantViolation(f"{attribute} 는 음수일 수 없다.", subject=self.name)

    @property
    def bytes_moved(self) -> int:
        return self.weight_bytes + self.activation_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """1바이트당 몇 번 계산하는가."""
        if self.bytes_moved == 0:
            return 0.0
        return self.mac_count / self.bytes_moved

    def bottleneck(self, machine_balance: float, *, tolerance: float = 0.25) -> BottleneckKind:
        if self.mac_count == 0 and self.bytes_moved == 0:
            return BottleneckKind.NEGLIGIBLE
        intensity = self.arithmetic_intensity
        if machine_balance <= 0:
            return BottleneckKind.BALANCED
        ratio = intensity / machine_balance
        if ratio < 1.0 - tolerance:
            return BottleneckKind.MEMORY_BOUND
        if ratio > 1.0 + tolerance:
            return BottleneckKind.COMPUTE_BOUND
        return BottleneckKind.BALANCED


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """디바이스가 낼 수 있는 성능.

    이 두 숫자는 데이터시트에서 온다. 모델이 정하는 것이 아니다.
    """

    name: str
    peak_gmac_per_second: float
    memory_bandwidth_gb_per_second: float

    def __post_init__(self) -> None:
        if self.peak_gmac_per_second <= 0:
            raise InvariantViolation(
                "연산 성능은 0보다 커야 한다.", subject="peak_gmac_per_second"
            )
        if self.memory_bandwidth_gb_per_second <= 0:
            raise InvariantViolation(
                "메모리 대역폭은 0보다 커야 한다.",
                subject="memory_bandwidth_gb_per_second",
            )

    @property
    def machine_balance(self) -> float:
        """MAC/byte. 이 값보다 산술 강도가 낮으면 메모리가 병목이다."""
        return self.peak_gmac_per_second / self.memory_bandwidth_gb_per_second

    def describe(self) -> str:
        return (
            f"{self.name}: {self.peak_gmac_per_second:g} GMAC/s, "
            f"{self.memory_bandwidth_gb_per_second:g} GB/s "
            f"→ 균형점 {self.machine_balance:.1f} MAC/byte"
        )


@dataclass(frozen=True, slots=True)
class RooflineProfile:
    """모델 전체의 비용 구조."""

    layers: tuple[LayerCost, ...] = field(default_factory=tuple)
    device: DeviceCapability | None = None

    @property
    def total_macs(self) -> int:
        return sum(layer.mac_count for layer in self.layers)

    @property
    def total_bytes_moved(self) -> int:
        return sum(layer.bytes_moved for layer in self.layers)

    @property
    def overall_intensity(self) -> float:
        if self.total_bytes_moved == 0:
            return 0.0
        return self.total_macs / self.total_bytes_moved

    @property
    def machine_balance(self) -> float:
        return self.device.machine_balance if self.device else 0.0

    def bottleneck_of(self, layer: LayerCost) -> BottleneckKind:
        return layer.bottleneck(self.machine_balance)

    @property
    def dominant_bottleneck(self) -> BottleneckKind:
        """연산량 기준으로 가장 큰 몫을 차지하는 병목."""
        if not self.layers:
            return BottleneckKind.NEGLIGIBLE
        weights: dict[BottleneckKind, int] = {}
        for layer in self.layers:
            kind = self.bottleneck_of(layer)
            weights[kind] = weights.get(kind, 0) + max(layer.mac_count, layer.bytes_moved)
        return max(weights.items(), key=lambda item: item[1])[0]

    @property
    def busiest_layer(self) -> LayerCost | None:
        if not self.layers:
            return None
        return max(self.layers, key=lambda layer: layer.mac_count)

    @property
    def heaviest_traffic_layer(self) -> LayerCost | None:
        if not self.layers:
            return None
        return max(self.layers, key=lambda layer: layer.bytes_moved)

    def render(self) -> str:
        lines = [
            f"{'layer':<20}{'MACs':>12}{'bytes':>12}{'intensity':>12}  병목",
            "-" * 70,
        ]
        for layer in self.layers:
            lines.append(
                f"{layer.name:<20}{layer.mac_count:>12,}{layer.bytes_moved:>12,}"
                f"{layer.arithmetic_intensity:>12.2f}  {self.bottleneck_of(layer).value}"
            )
        lines.append("-" * 70)
        lines.append(
            f"{'합계':<20}{self.total_macs:>12,}{self.total_bytes_moved:>12,}"
            f"{self.overall_intensity:>12.2f}"
        )
        if self.device:
            lines.append(f"  {self.device.describe()}")
            lines.append(f"  지배적 병목: {self.dominant_bottleneck.value}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RooflinePolicy:
    """병목 구조에 대한 판단."""

    max_memory_bound_share: float = 0.5
    """연산의 절반 이상이 메모리 병목이면 양자화가 잘 듣는다는 신호다."""

    min_intensity_warning: float = 1.0

    def inspect(self, profile: RooflineProfile) -> tuple[Finding, ...]:
        if not profile.layers or profile.device is None:
            return ()

        findings: list[Finding] = []
        memory_bound = [
            layer
            for layer in profile.layers
            if profile.bottleneck_of(layer) is BottleneckKind.MEMORY_BOUND
        ]
        total = profile.total_macs or 1
        share = sum(layer.mac_count for layer in memory_bound) / total

        if share > self.max_memory_bound_share:
            findings.append(
                Finding(
                    code="ROOFLINE_MEMORY_BOUND",
                    message=(
                        f"연산의 {share:.0%} 가 메모리 병목 구간이다. "
                        "연산을 줄여도 별로 안 빨라진다 — 데이터를 줄여야 한다. "
                        "양자화(FP16/INT8)가 잘 듣는 상황이다."
                    ),
                    severity=Severity.INFO,
                    subject="memory",
                    measured=share,
                    threshold=self.max_memory_bound_share,
                )
            )
        else:
            findings.append(
                Finding(
                    code="ROOFLINE_COMPUTE_BOUND",
                    message=(
                        "연산 병목이 우세하다. 양자화로 크기는 줄어도 "
                        "속도는 기대만큼 안 빨라질 수 있다."
                    ),
                    severity=Severity.INFO,
                    subject="compute",
                    measured=1.0 - share,
                    threshold=self.max_memory_bound_share,
                )
            )

        busiest = profile.busiest_layer
        if busiest is not None and busiest.mac_count > total * 0.5:
            findings.append(
                Finding(
                    code="ROOFLINE_SINGLE_HOTSPOT",
                    message=(
                        f"'{busiest.name}' 한 층이 전체 연산의 "
                        f"{busiest.mac_count / total:.0%} 를 차지한다. "
                        "여기를 고치지 않으면 나머지를 아무리 줄여도 소용없다."
                    ),
                    severity=Severity.WARNING,
                    subject=busiest.name,
                    measured=busiest.mac_count / total,
                    threshold=0.5,
                )
            )

        return tuple(findings)
