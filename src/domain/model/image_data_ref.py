"""이미지로 학습할 때의 데이터 참조와 그 게이트. (실습 3-11)

표(CSV)로 학습할 때의 `TrainingDataRef` 와 형제다. 다른 점은 하나다.

    TrainingDataRef  "어떤 열을, 어떻게 잘라서 볼 것인가"
    ImageDataRef     "어떤 폴더를, 어떤 크기로 볼 것인가"

폴더 이름이 곧 라벨이다. 그래서 **폴더를 잘못 만들면 라벨이 잘못된다** —
모듈 2 에서 본 라벨 오류가 이미지에서는 이런 모습으로 나타난다.

여기에도 게이트가 있다. 표의 게이트(모듈 1·2)와 검사 항목이 다르기 때문에
이미지에는 이미지의 게이트를 둔다. 재는 것은 Infrastructure 고, **판정은 여기서** 한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.model.tensor_spec import ImageTensorSpec
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict, derive_verdict


@dataclass(frozen=True, slots=True)
class ImageFolderReport:
    """이미지 폴더를 실제로 읽어 본 결과. Infrastructure 가 채운다.

    여기에 배열은 없다. 개수와 크기뿐이다.
    """

    root_uri: str
    class_counts: Mapping[str, int]
    unreadable_count: int = 0
    distinct_size_count: int = 1
    duplicate_count: int = 0

    def __post_init__(self) -> None:
        if any(count < 0 for count in self.class_counts.values()):
            raise InvariantViolation("장수는 음수일 수 없다.", subject="class_counts")
        if self.unreadable_count < 0:
            raise InvariantViolation(
                "읽지 못한 장수는 음수일 수 없다.", subject="unreadable_count"
            )

    @property
    def readable_count(self) -> int:
        return sum(self.class_counts.values())

    @property
    def total_count(self) -> int:
        return self.readable_count + self.unreadable_count

    @property
    def unreadable_ratio(self) -> float:
        return self.unreadable_count / self.total_count if self.total_count else 0.0

    @property
    def imbalance_ratio(self) -> float:
        """가장 많은 클래스 ÷ 가장 적은 클래스."""
        counts = [c for c in self.class_counts.values() if c > 0]
        if len(counts) < 2:
            return 1.0
        return max(counts) / min(counts)

    def describe(self) -> str:
        lines = [f"{self.root_uri}", f"  읽음 {self.readable_count:,}장"]
        for name, count in sorted(self.class_counts.items()):
            lines.append(f"    {name:<12} {count:>6,}장")
        if self.unreadable_count:
            lines.append(f"  못 읽음 {self.unreadable_count}장")
        if self.distinct_size_count > 1:
            lines.append(f"  해상도 {self.distinct_size_count} 종류")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ImageReadinessPolicy:
    """이미지 데이터로 학습을 시작해도 되는가. (실습 3-11)

    표 데이터의 게이트와 항목이 다르다.
    결측치도 시간축도 없다. 대신 **장수와 폴더와 해상도**가 있다.
    """

    min_samples_per_class: int = 30
    """클래스당 이 정도는 있어야 '학습했다'고 말할 수 있다."""

    max_unreadable_ratio: float = 0.05
    max_imbalance_ratio: float = 5.0

    def inspect(self, report: ImageFolderReport) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        # "클래스가 하나뿐"인 경우는 여기서 다루지 않는다.
        # 그건 소견이 아니라 **불변식 위반**이다 — ImageDataRef 가 스스로 막는다.
        present = {n: c for n, c in report.class_counts.items() if c > 0}

        for name, count in sorted(present.items()):
            if count < self.min_samples_per_class:
                findings.append(
                    Finding(
                        code="IMG_TOO_FEW_SAMPLES",
                        message=(
                            f"'{name}' 이 {count}장뿐이다. "
                            "이 수로 나온 정확도는 다음 주에 그대로 재현되지 않는다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=name,
                        measured=float(count),
                        threshold=float(self.min_samples_per_class),
                    )
                )

        if report.unreadable_ratio > self.max_unreadable_ratio:
            findings.append(
                Finding(
                    code="IMG_UNREADABLE",
                    message=(
                        f"{report.unreadable_count}장이 열리지 않는다. "
                        "**빠진 장은 대개 한쪽 클래스에 몰려 있다** — "
                        "그러면 빠진 것이 곧 편향이 된다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=report.root_uri,
                    measured=report.unreadable_ratio,
                    threshold=self.max_unreadable_ratio,
                )
            )
        elif report.unreadable_count:
            findings.append(
                Finding(
                    code="IMG_UNREADABLE_FEW",
                    message=(
                        f"{report.unreadable_count}장이 열리지 않는다. "
                        "학습에서 빠졌다는 사실을 기록에 남겨야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject=report.root_uri,
                    measured=float(report.unreadable_count),
                )
            )

        if report.imbalance_ratio > self.max_imbalance_ratio:
            findings.append(
                Finding(
                    code="IMG_IMBALANCED",
                    message=(
                        f"많은 쪽이 적은 쪽의 {report.imbalance_ratio:.1f}배다. "
                        "많은 쪽만 찍어도 정확도가 높게 나온다."
                    ),
                    severity=Severity.WARNING,
                    subject=report.root_uri,
                    measured=report.imbalance_ratio,
                    threshold=self.max_imbalance_ratio,
                )
            )

        if report.distinct_size_count > 1:
            findings.append(
                Finding(
                    code="IMG_MIXED_RESOLUTION",
                    message=(
                        f"해상도가 {report.distinct_size_count} 종류다. "
                        "리사이즈로 모양은 맞지만 **가늘고 작은 결함은 크기마다 다르게 뭉개진다.**"
                    ),
                    severity=Severity.WARNING,
                    subject=report.root_uri,
                    measured=float(report.distinct_size_count),
                    threshold=1.0,
                )
            )

        if report.duplicate_count:
            findings.append(
                Finding(
                    code="IMG_DUPLICATE",
                    message=(
                        f"같은 이미지가 {report.duplicate_count}쌍 있다. "
                        "분할 양쪽에 나뉘어 들어가면 시험지가 유출된다."
                    ),
                    severity=Severity.WARNING,
                    subject=report.root_uri,
                    measured=float(report.duplicate_count),
                )
            )

        return tuple(findings)


@dataclass(frozen=True, slots=True)
class ImageDataRef:
    """이미지로 학습할 데이터 한 덩어리 (Anti-Corruption Layer VO)."""

    dataset_ref: str
    root_uri: str
    spec: ImageTensorSpec
    class_labels: tuple[str, ...]
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15)

    readiness_findings: tuple[Finding, ...] = field(default_factory=tuple)
    """이미지 게이트가 남긴 소견. 판정은 여기서 유도한다."""

    def __post_init__(self) -> None:
        if not self.dataset_ref.strip():
            raise InvariantViolation(
                "어느 Dataset 으로 학습했는지 없으면 모델을 되돌릴 수 없다.",
                subject="dataset_ref",
            )
        if not self.root_uri.strip():
            raise InvariantViolation("이미지 위치가 없다.", subject="root_uri")
        if len(self.class_labels) < 2:
            raise InvariantViolation(
                "클래스가 둘 미만이면 분류 문제가 아니다.", subject="class_labels"
            )
        if len(set(self.class_labels)) != len(self.class_labels):
            raise InvariantViolation(
                "같은 라벨이 두 번 있다.", subject="class_labels"
            )
        total = sum(self.split_ratio)
        if abs(total - 1.0) > 1e-9:
            raise InvariantViolation(
                f"분할 비율의 합이 {total:.4f} 다. 1.0 이어야 한다.",
                subject="split_ratio",
            )
        if any(r <= 0 for r in self.split_ratio):
            raise InvariantViolation("모든 분할은 0보다 커야 한다.", subject="split_ratio")

    @property
    def class_count(self) -> int:
        return len(self.class_labels)

    @property
    def verdict(self) -> Verdict:
        return derive_verdict(self.readiness_findings)

    @property
    def gates_passed(self) -> bool:
        return self.verdict is not Verdict.FAILED

    @property
    def missing_gates(self) -> tuple[str, ...]:
        return tuple(
            f.describe() for f in self.readiness_findings if f.is_blocking
        )
