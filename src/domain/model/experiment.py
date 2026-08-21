"""실험을 나란히 놓고 비교한다. (실습 3-12, 3-14, 3-15, 6-12)

모델을 한 번 만들어서 끝나는 일은 없다. 스무 번쯤 만든다.
그러면 두 달 뒤에 반드시 이 질문이 온다.

    "그때 그 0.91 짜리, 어떻게 만든 거였죠?"

기록이 없으면 대답은 "다시 해 봐야 압니다"다.
그래서 실험은 **결과만이 아니라 조건과 함께** 남겨야 한다.

여기서 지키는 규칙은 하나다.

    **한 번에 하나만 바꾼다.**

두 개를 같이 바꾸고 좋아지면, 무엇 때문에 좋아졌는지 알 수 없다.
그 상태로 다음 실험을 하면 잘못된 방향으로 스무 번을 간다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class TrialKnobs:
    """이 시행에서 무엇을 어떻게 두었는가.

    "학습률 0.003" 같은 것만이 아니다.
    **창 길이도, 입력 열도, 데이터 자체도 손잡이다.**
    데이터를 바꿔 놓고 손잡이에 적지 않으면, 모델을 비교했다고 착각하게 된다.
    """

    values: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.values:
            raise InvariantViolation(
                "무엇을 어떻게 두었는지 없으면 재현할 수 없다.", subject="values"
            )

    def differences(self, other: TrialKnobs) -> tuple[str, ...]:
        """두 시행에서 다르게 둔 손잡이 이름."""
        names = set(self.values) | set(other.values)
        return tuple(
            sorted(
                name
                for name in names
                if self.values.get(name) != other.values.get(name)
            )
        )

    def describe(self) -> str:
        return " ".join(f"{k}={v}" for k, v in sorted(self.values.items()))


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    """시행 하나가 남긴 숫자.

    정확도 하나만 적지 않는다. **정확도만 적으면 정확도만 좋아진다.**
    """

    accuracy: float
    macro_recall: float
    macro_f1: float
    loss: float
    latency_ms_p50: float = 0.0
    parameter_count: int = 0
    epochs: int = 0
    evaluated_samples: int = 0
    """이 숫자를 몇 개로 재었는가.

    **지표는 표본 수와 함께 읽어야 한다.** 16개로 잰 1.000 과
    900개로 잰 0.97 중 믿을 수 있는 것은 뒤쪽이다.
    """

    def __post_init__(self) -> None:
        for name in ("accuracy", "macro_recall", "macro_f1"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    def value_of(self, metric: str) -> float:
        try:
            return float(getattr(self, metric))
        except AttributeError as exc:  # pragma: no cover - 방어
            raise InvariantViolation(
                f"'{metric}' 라는 지표는 없다.", subject="metric"
            ) from exc


@dataclass(frozen=True, slots=True)
class ExperimentTrial:
    """실험 한 번."""

    label: str
    knobs: TrialKnobs
    metrics: TrialMetrics
    seed: int
    data_ref: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise InvariantViolation("시행에 이름이 없다.", subject="label")
        if not self.data_ref.strip():
            raise InvariantViolation(
                "어떤 데이터로 한 실험인지 없으면 비교할 수 없다.", subject="data_ref"
            )


@dataclass(frozen=True, slots=True)
class ExperimentBoard:
    """시행들을 나란히 놓은 표.

    표를 만드는 것이 목적이 아니다. **표가 있어야 고르는 근거가 생긴다.**
    """

    name: str
    trials: tuple[ExperimentTrial, ...] = field(default_factory=tuple)

    def with_trial(self, trial: ExperimentTrial) -> ExperimentBoard:
        if any(t.label == trial.label for t in self.trials):
            raise InvariantViolation(
                f"'{trial.label}' 이라는 시행이 이미 있다. "
                "같은 이름을 두 번 쓰면 둘 중 어느 것인지 알 수 없다.",
                subject="label",
            )
        return ExperimentBoard(name=self.name, trials=(*self.trials, trial))

    @property
    def is_empty(self) -> bool:
        return not self.trials

    def best_by(self, metric: str = "macro_f1") -> ExperimentTrial:
        if not self.trials:
            raise InvariantViolation(
                "시행이 하나도 없다.", subject=self.name
            )
        return max(
            self.trials,
            key=lambda t: (t.metrics.value_of(metric), -t.metrics.latency_ms_p50),
        )

    def spread_of(self, metric: str = "macro_f1") -> float:
        """1등과 꼴등의 차이. 이 폭이 좁으면 무엇을 골라도 비슷하다."""
        if len(self.trials) < 2:
            return 0.0
        values = [t.metrics.value_of(metric) for t in self.trials]
        return max(values) - min(values)

    def gap_to_runner_up(self, metric: str = "macro_f1") -> float:
        if len(self.trials) < 2:
            return 0.0
        values = sorted(
            (t.metrics.value_of(metric) for t in self.trials), reverse=True
        )
        return values[0] - values[1]

    def render(self, metric: str = "macro_f1") -> str:
        header = (
            f"{'시행':<22}{'accuracy':>10}{'macroF1':>10}{'recall':>9}"
            f"{'loss':>9}{'표본':>8}{'params':>10}"
        )
        lines = [f"[{self.name}]", header, "-" * len(header)]
        best = self.best_by(metric) if self.trials else None
        for trial in self.trials:
            m = trial.metrics
            mark = " ←" if best is not None and trial.label == best.label else ""
            lines.append(
                f"{trial.label:<22}{m.accuracy:>10.3f}{m.macro_f1:>10.3f}"
                f"{m.macro_recall:>9.3f}{m.loss:>9.3f}{m.evaluated_samples:>8,}"
                f"{m.parameter_count:>10,}{mark}"
            )
        lines.append("-" * len(header))
        for trial in self.trials:
            lines.append(f"  {trial.label:<20} {trial.knobs.describe()}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ExperimentPolicy:
    """이 비교를 믿어도 되는가. (실습 3-14)

    비교표가 있다고 비교가 된 것은 아니다.
    """

    noise_band: float = 0.02
    """이만큼 안쪽의 차이는 **차이가 아니다.**

    같은 데이터를 시드만 바꿔 두 번 학습해 보면 이 정도는 그냥 흔들린다.
    그 폭 안에서 1등을 고르는 것은 동전을 던지는 것과 같다.
    """

    min_evaluated_samples: int = 60
    """이보다 적은 표본으로 잰 지표는 비교의 근거가 되지 못한다."""

    def inspect(
        self, board: ExperimentBoard, *, metric: str = "macro_f1"
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        for trial in board.trials:
            count = trial.metrics.evaluated_samples
            if 0 < count < self.min_evaluated_samples:
                findings.append(
                    Finding(
                        code="EXP_TOO_FEW_EVALUATED",
                        message=(
                            f"'{trial.label}' 의 지표는 {count}개로 잰 것이다. "
                            f"**{trial.metrics.accuracy:.3f} 이라는 숫자가 "
                            "한두 개 차이로 뒤집힌다.** 표본이 이만큼이면 "
                            "비교의 근거가 되지 못한다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=trial.label,
                        measured=float(count),
                        threshold=float(self.min_evaluated_samples),
                    )
                )

        if len(board.trials) < 2:
            findings.append(
                Finding(
                    code="EXP_SINGLE_TRIAL",
                    message=(
                        "시행이 하나뿐이다. 비교가 아니라 기록이다. "
                        "**기준선 없는 숫자는 좋은지 나쁜지 말할 수 없다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=board.name,
                    measured=float(len(board.trials)),
                    threshold=2.0,
                )
            )
            return tuple(findings)

        # 연속한 두 시행에서 손잡이를 몇 개 바꿨는가
        for previous, current in zip(board.trials, board.trials[1:], strict=False):
            changed = previous.knobs.differences(current.knobs)
            if len(changed) > 1:
                findings.append(
                    Finding(
                        code="EXP_MULTIPLE_KNOBS_CHANGED",
                        message=(
                            f"'{previous.label}' → '{current.label}' 에서 "
                            f"{len(changed)}개를 한꺼번에 바꿨다 ({', '.join(changed)}). "
                            "좋아져도 **무엇 때문인지 알 수 없다.**"
                        ),
                        severity=Severity.WARNING,
                        subject=current.label,
                        measured=float(len(changed)),
                        threshold=1.0,
                    )
                )

        seeds = {t.seed for t in board.trials}
        declares_seed = any("seed" in t.knobs.values for t in board.trials)
        if len(seeds) > 1 and not declares_seed:
            findings.append(
                Finding(
                    code="EXP_SEEDS_DIFFER",
                    message=(
                        f"시드가 {sorted(seeds)} 로 서로 다른데 손잡이에 적혀 있지 않다. "
                        "차이가 구조 때문인지 **운 때문인지 구분할 수 없다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=board.name,
                    measured=float(len(seeds)),
                    threshold=1.0,
                )
            )

        data_refs = {t.data_ref for t in board.trials}
        declares_data = any("data" in t.knobs.values for t in board.trials)
        if len(data_refs) > 1 and not declares_data:
            findings.append(
                Finding(
                    code="EXP_DATA_DIFFERS",
                    message=(
                        f"데이터가 {len(data_refs)}종류인데 손잡이에 적혀 있지 않다. "
                        "**모델을 비교한 것이 아니라 데이터를 비교한 것이다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=board.name,
                    measured=float(len(data_refs)),
                    threshold=1.0,
                )
            )

        gap = board.gap_to_runner_up(metric)
        if gap < self.noise_band:
            best = board.best_by(metric)
            findings.append(
                Finding(
                    code="EXP_WITHIN_NOISE",
                    message=(
                        f"1등('{best.label}')과 2등의 차이가 {gap:.4f} 다. "
                        f"흔들림 폭({self.noise_band}) 안이다 — "
                        "**이 차이로 결론을 내면 안 된다.** "
                        "시드를 바꿔 몇 번 더 돌려 보고 정한다."
                    ),
                    severity=Severity.WARNING,
                    subject=best.label,
                    measured=gap,
                    threshold=self.noise_band,
                )
            )

        return tuple(findings)
