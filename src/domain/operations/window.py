"""관측 창.

현장 지표는 **점이 아니라 구간**으로 잰다.
추론 하나의 지연시간은 아무 뜻도 없다. 한 시간 치의 p95 가 뜻이 있다.

그리고 창이 있어야 '언제부터'에 답할 수 있다 (실습 5-4).
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    """관측 구간 하나."""

    label: str
    started_at: str
    ended_at: str
    sample_count: int
    device_id: str | None = None
    """한 대만 본 창이면 여기에 적는다. 전체 창이면 비어 있다."""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise InvariantViolation("창에 이름이 없다.", subject="label")
        if not self.started_at.strip() or not self.ended_at.strip():
            raise InvariantViolation("창의 시작과 끝이 필요하다.", subject="range")
        if self.started_at > self.ended_at:
            raise InvariantViolation(
                "끝이 시작보다 앞이다.", subject="range"
            )
        if self.sample_count < 0:
            raise InvariantViolation("음수 표본은 없다.", subject="sample_count")

    @property
    def is_device_scoped(self) -> bool:
        return self.device_id is not None

    def describe(self) -> str:
        scope = f" [{self.device_id}]" if self.device_id else ""
        return f"{self.label}{scope} ({self.sample_count:,}건)"


@dataclass(frozen=True, slots=True)
class WindowPolicy:
    """이 창으로 판단해도 되는가.

    표본이 적은 창의 p95 는 다음에 재면 달라진다.
    그 숫자로 배포를 격리하면 멀쩡한 모델을 내린다.
    """

    min_sample_count: int = 100

    def inspect(self, window: ObservationWindow) -> tuple[str, ...]:
        if window.sample_count < self.min_sample_count:
            return (
                f"표본 {window.sample_count}건으로 판단하고 있다. "
                f"{self.min_sample_count}건은 있어야 이 창의 숫자를 믿을 수 있다.",
            )
        return ()

    def is_reliable(self, window: ObservationWindow) -> bool:
        return window.sample_count >= self.min_sample_count
