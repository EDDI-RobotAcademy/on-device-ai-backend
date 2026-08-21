"""Data Drift를 직접 찾아라. (실습 5-7)

실습 1-9 에서 이미 PSI 를 봤다. 그때는 **학습 전에** 물었다.

    "이 학습 데이터가 현실을 대표하는가?"

지금은 **학습 후에** 묻는다.

    "현실이 학습 데이터에서 얼마나 멀어졌는가?"

같은 계산, 다른 질문이다. 그리고 이쪽이 훨씬 중요하다 —
데이터는 고정돼 있는데 현실은 계속 움직이기 때문이다.

세 가지를 구분한다.

    입력 드리프트   들어오는 값이 변했다        ← 여기
    예측 드리프트   나가는 답이 변했다          ← 실습 5-6
    개념 드리프트   입력과 정답의 관계가 변했다 ← **정답 없이는 못 잰다**

세 번째를 못 잰다는 사실이 중요하다. 그래서 앞의 둘로 대신 본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.operations.window import ObservationWindow
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """입력 채널 하나가 학습 때와 얼마나 달라졌는가."""

    field_name: str
    psi: float
    """Population Stability Index. 실습 1-9 와 같은 계산이다."""

    mean_shift_sigma: float
    """평균이 학습 때 표준편차의 몇 배만큼 이동했는가."""

    out_of_range_ratio: float
    """학습 때 본 적 없는 범위로 들어온 표본의 비율."""

    def __post_init__(self) -> None:
        for name in ("psi", "out_of_range_ratio"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        if not 0.0 <= self.out_of_range_ratio <= 1.0:
            raise InvariantViolation(
                "out_of_range_ratio 는 0~1 이어야 한다.", subject="out_of_range_ratio"
            )

    @property
    def severity_rank(self) -> int:
        """PSI 관례. 0.1 미만은 무시, 0.2 넘으면 조치."""
        if self.psi >= 0.2:
            return 2
        if self.psi >= 0.1:
            return 1
        return 0

    def describe(self) -> str:
        return (
            f"{self.field_name:<18}PSI {self.psi:>7.4f}  "
            f"평균이동 {self.mean_shift_sigma:>+6.2f}σ  "
            f"범위밖 {self.out_of_range_ratio:>6.1%}"
        )


@dataclass(frozen=True, slots=True)
class DriftReport:
    """한 창의 입력 드리프트."""

    window: ObservationWindow
    features: tuple[FeatureDrift, ...] = field(default_factory=tuple)

    @property
    def worst(self) -> FeatureDrift | None:
        return max(self.features, key=lambda f: f.psi) if self.features else None

    @property
    def max_psi(self) -> float:
        worst = self.worst
        return worst.psi if worst else 0.0

    def drifted(self, threshold: float = 0.2) -> tuple[FeatureDrift, ...]:
        return tuple(f for f in self.features if f.psi >= threshold)

    def out_of_range(self, threshold: float = 0.01) -> tuple[FeatureDrift, ...]:
        return tuple(
            f for f in self.features if f.out_of_range_ratio >= threshold
        )

    def render(self) -> str:
        lines = [f"입력 드리프트 — {self.window.describe()}", "-" * 62]
        lines += [
            f"  {f.describe()}"
            for f in sorted(self.features, key=lambda f: -f.psi)
        ]
        lines.append("-" * 62)
        lines.append(f"  최대 PSI {self.max_psi:.4f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DriftPolicy:
    """어디부터 드리프트라고 부를 것인가.

    이 숫자들은 통계가 정하지 않는다. **재학습 비용이 정한다.**
    라벨링에 2주가 걸리는 현장이면 기준을 낮게 잡아 미리 시작해야 한다.
    """

    max_psi: float = 0.2
    watch_psi: float = 0.1
    max_out_of_range_ratio: float = 0.01
    max_drifted_field_count: int = 2

    def inspect(self, report: DriftReport) -> tuple[Finding, ...]:
        if not report.features:
            return (
                Finding(
                    code="OPS_NO_DRIFT_BASELINE",
                    message=(
                        "입력 분포를 비교할 기준이 없다. "
                        "학습 데이터의 분포를 남겨 두지 않으면 드리프트를 잴 수 없다."
                    ),
                    severity=Severity.WARNING,
                    subject=report.window.label,
                ),
            )

        findings: list[Finding] = []
        drifted = report.drifted(self.max_psi)

        for feature in drifted:
            findings.append(
                Finding(
                    code="OPS_INPUT_DRIFT",
                    message=(
                        f"'{feature.field_name}' 의 분포가 학습 때와 크게 다르다 "
                        f"(평균 {feature.mean_shift_sigma:+.2f}σ 이동). "
                        "모델은 이 값을 본 적이 별로 없다."
                    ),
                    severity=Severity.WARNING,
                    subject=feature.field_name,
                    measured=feature.psi,
                    threshold=self.max_psi,
                )
            )

        watching = [
            f
            for f in report.features
            if self.watch_psi <= f.psi < self.max_psi
        ]
        for feature in watching:
            findings.append(
                Finding(
                    code="OPS_INPUT_DRIFT_WATCH",
                    message=(
                        f"'{feature.field_name}' 이 움직이기 시작했다. "
                        "아직 조치할 정도는 아니지만 다음 창을 봐야 한다."
                    ),
                    severity=Severity.INFO,
                    subject=feature.field_name,
                    measured=feature.psi,
                    threshold=self.max_psi,
                )
            )

        for feature in report.out_of_range(self.max_out_of_range_ratio):
            findings.append(
                Finding(
                    code="OPS_INPUT_OUT_OF_RANGE",
                    message=(
                        f"'{feature.field_name}' 이 학습 때 본 적 없는 범위로 들어온다. "
                        "모델은 이 구간에서 무슨 답을 할지 아무도 모른다 — "
                        "틀리는 게 아니라 **정의되지 않는다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=feature.field_name,
                    measured=feature.out_of_range_ratio,
                    threshold=self.max_out_of_range_ratio,
                )
            )

        if len(drifted) > self.max_drifted_field_count:
            findings.append(
                Finding(
                    code="OPS_MULTI_FEATURE_DRIFT",
                    message=(
                        f"{len(drifted)}개 채널이 동시에 움직였다. "
                        "센서 하나가 고장 난 게 아니라 **공정 자체가 바뀐 것**에 가깝다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=report.window.label,
                    measured=float(len(drifted)),
                    threshold=float(self.max_drifted_field_count),
                )
            )

        return tuple(findings)
