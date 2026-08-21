"""Cloud에서 다시 학습 데이터를 만들어라. (실습 6-4)

**여기가 순환이 닫히는 지점이다.** 모듈 1 로 돌아간다.

그런데 올라온 것을 전부 쓰면 안 된다.
현장 데이터에는 학습에 넣으면 안 되는 것이 섞여 있다.

    격리된 디바이스의 데이터    이상한 상태에서 낸 판단이다 (실습 5-8)
    드리프트 이전 구간          지금 고치려는 그 문제가 없던 시절이다 (실습 5-7)
    라벨이 없는 구간            학습시킬 것이 없다 (실습 5-3)
    한 디바이스에 몰린 데이터   그 디바이스의 버릇을 학습한다

마지막 것이 특히 자주 놓친다.
DEV-02 가 다른 대보다 열 배 많이 올렸다면, 새 모델은 DEV-02 를 학습한다.

그리고 이 명세는 **모듈 1 의 Dataset 이 아니다.**
어디서 무엇을 골랐는지에 대한 기록이고, 그 결과물이 모듈 1 의 입력이 된다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict, derive_verdict


@dataclass(frozen=True, slots=True)
class SourceWindow:
    """어느 구간을 쓸 것인가."""

    started_at: str
    ended_at: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.started_at > self.ended_at:
            raise InvariantViolation("끝이 시작보다 앞이다.", subject="window")

    def contains(self, moment: str) -> bool:
        return self.started_at <= moment <= self.ended_at

    def describe(self) -> str:
        note = f" — {self.reason}" if self.reason else ""
        return f"{self.started_at} ~ {self.ended_at}{note}"


@dataclass(frozen=True, slots=True)
class DatasetBuildSpec:
    """현장 데이터로 학습 데이터를 만드는 계획."""

    build_id: str
    window: SourceWindow
    device_ids: tuple[str, ...]
    excluded_devices: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """뺀 디바이스와 그 이유. **뺐다는 사실이 기록으로 남아야 한다.**"""

    record_counts: Mapping[str, int] = field(default_factory=dict)
    labeled_counts: Mapping[str, int] = field(default_factory=dict)
    label_distribution: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device_ids:
            raise InvariantViolation(
                "디바이스를 하나도 안 골랐다.", subject="device_ids"
            )

    @property
    def total_records(self) -> int:
        return sum(self.record_counts.values())

    @property
    def total_labeled(self) -> int:
        return sum(self.labeled_counts.values())

    @property
    def labeled_ratio(self) -> float:
        return self.total_labeled / self.total_records if self.total_records else 0.0

    @property
    def device_share(self) -> dict[str, float]:
        total = self.total_records
        if not total:
            return {}
        return {d: n / total for d, n in self.record_counts.items()}

    @property
    def dominant_device(self) -> tuple[str, float]:
        share = self.device_share
        if not share:
            return "", 0.0
        device = max(share, key=lambda d: share[d])
        return device, share[device]

    @property
    def minority_label(self) -> tuple[str, int]:
        if not self.label_distribution:
            return "", 0
        label = min(self.label_distribution, key=lambda x: self.label_distribution[x])
        return label, self.label_distribution[label]

    def render(self) -> str:
        lines = [
            f"학습 데이터셋 계획 {self.build_id}",
            f"  구간   : {self.window.describe()}",
            f"  디바이스: {len(self.device_ids)}대  "
            f"(제외 {len(self.excluded_devices)}대)",
            f"  표본   : {self.total_records:,}건  "
            f"라벨 {self.total_labeled:,}건 ({self.labeled_ratio:.1%})",
        ]
        if self.label_distribution:
            lines.append(
                "  라벨   : "
                + "  ".join(
                    f"{k} {v:,}" for k, v in sorted(self.label_distribution.items())
                )
            )
        if self.excluded_devices:
            lines.append("")
            lines.append("  제외한 디바이스:")
            lines += [
                f"    ✗ {device}: {reason}" for device, reason in self.excluded_devices
            ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DatasetBuildCheck:
    """이 데이터로 학습해도 되는가."""

    spec: DatasetBuildSpec
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.findings)

    @property
    def can_build(self) -> bool:
        return self.verdict is not Verdict.FAILED

    def render(self) -> str:
        lines = [self.spec.render(), "", f"판정: {self.verdict.value}"]
        if self.findings:
            lines += [f"  - {f.describe()}" for f in self.findings]
        else:
            lines.append("  걸리는 것이 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DatasetBuildPolicy:
    """현장 데이터를 학습 데이터로 만들 때의 기준.

    모듈 1 의 판정과 겹치지 않는다. 여기서 보는 것은 **무엇을 골랐는가**이고,
    고른 뒤의 데이터 품질은 모듈 1·2 가 다시 본다.
    """

    min_records: int = 5_000
    min_labeled: int = 500
    min_labels_per_class: int = 30
    max_device_share: float = 0.5
    """한 디바이스가 이 비율을 넘으면 그 디바이스를 학습하는 것이다."""

    min_devices: int = 2
    require_exclusion_reasons: bool = True

    def inspect(self, spec: DatasetBuildSpec) -> DatasetBuildCheck:
        findings: list[Finding] = []

        if spec.total_records < self.min_records:
            findings.append(
                Finding(
                    code="BUILD_TOO_FEW_RECORDS",
                    message=(
                        f"{spec.total_records:,}건으로 학습하려 한다. "
                        "이 정도면 이전 모델이 배운 것을 다시 배우고 끝난다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=spec.build_id,
                    measured=float(spec.total_records),
                    threshold=float(self.min_records),
                )
            )

        if spec.total_labeled < self.min_labeled:
            findings.append(
                Finding(
                    code="BUILD_TOO_FEW_LABELS",
                    message=(
                        f"라벨이 {spec.total_labeled:,}건뿐이다 "
                        f"({spec.labeled_ratio:.1%}). "
                        "**현장 데이터가 많은 것과 학습할 수 있는 것은 다르다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject=spec.build_id,
                    measured=float(spec.total_labeled),
                    threshold=float(self.min_labeled),
                )
            )

        label, count = spec.minority_label
        if label and count < self.min_labels_per_class:
            findings.append(
                Finding(
                    code="BUILD_MINORITY_LABEL_STARVED",
                    message=(
                        f"'{label}' 라벨이 {count}건뿐이다. "
                        "이 클래스는 재학습해도 나아지지 않는다 (실습 5-11)."
                    ),
                    severity=Severity.CRITICAL,
                    subject=label,
                    measured=float(count),
                    threshold=float(self.min_labels_per_class),
                )
            )

        device, share = spec.dominant_device
        if device and share > self.max_device_share:
            findings.append(
                Finding(
                    code="BUILD_DEVICE_DOMINATED",
                    message=(
                        f"'{device}' 한 대가 전체의 {share:.1%} 를 차지한다. "
                        "**새 모델은 그 디바이스의 버릇을 학습한다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=device,
                    measured=share,
                    threshold=self.max_device_share,
                )
            )

        if len(spec.device_ids) < self.min_devices:
            findings.append(
                Finding(
                    code="BUILD_TOO_FEW_DEVICES",
                    message=(
                        f"{len(spec.device_ids)}대에서만 모았다. "
                        "설비마다 조금씩 다르다 — 한 대만 보면 그 한 대에 맞춘다."
                    ),
                    severity=Severity.WARNING,
                    subject=spec.build_id,
                    measured=float(len(spec.device_ids)),
                    threshold=float(self.min_devices),
                )
            )

        if self.require_exclusion_reasons:
            blank = [d for d, reason in spec.excluded_devices if not reason.strip()]
            if blank:
                findings.append(
                    Finding(
                        code="BUILD_UNEXPLAINED_EXCLUSION",
                        message=(
                            f"{blank} 를 이유 없이 뺐다. "
                            "**무엇을 뺐는지가 무엇을 넣었는지만큼 중요하다.**"
                        ),
                        severity=Severity.CRITICAL,
                        subject="excluded_devices",
                    )
                )

        return DatasetBuildCheck(spec=spec, findings=tuple(findings))
