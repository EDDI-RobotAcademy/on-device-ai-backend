"""OptimizationRun — Optimization Context 의 Aggregate Root.

최적화는 **후보를 모으고 고르는 일**이다. 한 번에 답이 나오지 않는다.

    기준 측정  →  후보 추가 (경로 × 정밀도)  →  한 표에 놓기  →  선택

지키는 불변식:
    - 승인받지 않은 모델은 최적화 대상이 아니다 (모듈 3 게이트)
    - 기준을 재기 전에는 후보를 비교할 수 없다
    - 같은 (런타임, 정밀도) 조합은 하나만 남는다 — 다시 만들면 덮어쓴다
    - 선택된 뒤에는 후보를 몰래 추가할 수 없다
"""

from __future__ import annotations

from enum import Enum

from domain.optimization import events as domain_events
from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.benchmark import BenchmarkResult
from domain.optimization.conversion import ConversionRecord
from domain.optimization.errors import NoCandidateSelected
from domain.optimization.identifiers import OptimizationRunId
from domain.optimization.roofline import RooflineProfile
from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
from domain.optimization.selection import OptimizationCertificate, SelectionPolicy
from domain.optimization.tradeoff import (
    ArtifactAccuracy,
    OptimizationCandidate,
    TradeoffTable,
)
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.events import EventRecorder


class OptimizationStatus(Enum):
    OPEN = "OPEN"
    """시작했다. 기준도 아직 재지 않았다."""

    BENCHMARKED = "BENCHMARKED"
    """기준을 쟀다. 이제 후보를 비교할 수 있다."""

    EXPLORING = "EXPLORING"
    """후보를 모으는 중이다."""

    SELECTED = "SELECTED"
    BLOCKED = "BLOCKED"
    """예산을 만족하는 후보가 하나도 없다."""


class OptimizationRun(EventRecorder):
    """한 모델에 대한 최적화 시도."""

    __slots__ = (
        "_id",
        "_baseline",
        "_status",
        "_baseline_candidate",
        "_candidates",
        "_rejections",
        "_roofline",
        "_certificate",
    )

    def __init__(self, run_id: OptimizationRunId, baseline: BaselineModelRef) -> None:
        super().__init__()
        self._id = run_id
        self._baseline = baseline
        self._status = OptimizationStatus.OPEN
        self._baseline_candidate: OptimizationCandidate | None = None
        self._candidates: dict[str, OptimizationCandidate] = {}
        self._rejections: list[tuple[str, str]] = []
        self._roofline: RooflineProfile | None = None
        self._certificate: OptimizationCertificate | None = None

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def start(
        cls,
        run_id: OptimizationRunId,
        baseline: BaselineModelRef,
        *,
        require_accepted: bool = True,
    ) -> OptimizationRun:
        """최적화를 시작한다.

        승인받지 않은 모델이면 여기서 멈춘다.
        아직 쓸 수 있는지도 모르는 모델을 빠르게 만드는 것은 순서가 틀린 일이다.
        """
        if require_accepted and not baseline.accepted:
            raise IllegalStateTransition(
                "승인받지 않은 모델은 최적화 대상이 아니다: "
                + ", ".join(baseline.missing_gates),
                subject=str(run_id),
            )
        run = cls(run_id, baseline)
        run._record(
            domain_events.OptimizationRunStarted(
                run_id=run_id, model_version_id=baseline.model_version_id
            )
        )
        return run

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> OptimizationRunId:
        return self._id

    @property
    def baseline(self) -> BaselineModelRef:
        return self._baseline

    @property
    def status(self) -> OptimizationStatus:
        return self._status

    @property
    def baseline_candidate(self) -> OptimizationCandidate | None:
        return self._baseline_candidate

    @property
    def candidates(self) -> tuple[OptimizationCandidate, ...]:
        return tuple(self._candidates.values())

    def candidate_of(
        self, runtime: RuntimeTarget, precision: Precision
    ) -> OptimizationCandidate | None:
        return self._candidates.get(f"{runtime.value}/{precision.value}")

    @property
    def rejections(self) -> tuple[tuple[str, str], ...]:
        """변환 자체가 실패한 것들. 실패도 기록되어야 한다."""
        return tuple(self._rejections)

    @property
    def roofline(self) -> RooflineProfile | None:
        return self._roofline

    @property
    def certificate(self) -> OptimizationCertificate | None:
        return self._certificate

    @property
    def is_selected(self) -> bool:
        return self._status is OptimizationStatus.SELECTED

    def tradeoff_table(self) -> TradeoffTable:
        if self._baseline_candidate is None:
            raise IllegalStateTransition(
                "기준을 재기 전에는 후보를 비교할 수 없다.", subject=str(self._id)
            )
        return TradeoffTable(
            baseline=self._baseline_candidate, candidates=self.candidates
        )

    # -- 행위 --------------------------------------------------------------
    def record_baseline(
        self,
        artifact: ModelArtifact,
        benchmark: BenchmarkResult,
        accuracy: ArtifactAccuracy,
    ) -> None:
        """기준 모델을 재 둔다. (실습 4-1)

        이 숫자가 없으면 '빨라졌다'도 '떨어졌다'도 말할 수 없다.
        """
        self._guard_mutable("기준 측정")
        self._baseline_candidate = OptimizationCandidate(
            artifact=artifact,
            conversion=ConversionRecord(
                source_runtime=artifact.runtime,
                target_runtime=artifact.runtime,
                precision=artifact.precision,
                note="기준 모델 — 변환 없음",
            ),
            benchmark=benchmark,
            accuracy=accuracy,
        )
        if self._status is OptimizationStatus.OPEN:
            self._status = OptimizationStatus.BENCHMARKED
        self._record(
            domain_events.BaselineBenchmarked(
                run_id=self._id,
                p95_ms=benchmark.p95_ms,
                size_bytes=artifact.size_bytes,
            )
        )

    def attach_roofline(self, profile: RooflineProfile) -> None:
        """병목 구조를 붙인다. (실습 4-9)"""
        self._guard_mutable("병목 프로파일 기록")
        self._roofline = profile

    def add_candidate(self, candidate: OptimizationCandidate) -> None:
        """후보를 하나 추가한다. (실습 4-2 ~ 4-7)"""
        self._guard_mutable("후보 추가")
        if self._baseline_candidate is None:
            raise IllegalStateTransition(
                "기준을 재기 전에는 후보를 추가할 수 없다. "
                "비교할 대상이 없으면 그 숫자는 아무 뜻도 없다.",
                subject=str(self._id),
            )
        if candidate.artifact.parameter_count and (
            candidate.artifact.parameter_count != self._baseline.parameter_count
        ):
            raise InvariantViolation(
                f"후보의 파라미터 수({candidate.artifact.parameter_count})가 "
                f"기준({self._baseline.parameter_count})과 다르다. "
                "다른 모델을 비교하고 있는 것이다.",
                subject=candidate.label,
            )

        self._candidates[candidate.label] = candidate
        self._status = OptimizationStatus.EXPLORING
        self._record(
            domain_events.CandidateAdded(
                run_id=self._id,
                artifact_id=candidate.artifact.artifact_id,
                label=candidate.label,
                p95_ms=candidate.benchmark.p95_ms,
                size_bytes=candidate.artifact.size_bytes,
                accuracy=candidate.accuracy.accuracy,
            )
        )

    def record_rejection(self, label: str, reason: str) -> None:
        """변환이 실패했다는 사실을 남긴다.

        "TFLite 로는 안 되더라"가 구전으로만 남으면 다음 사람이 또 해 본다.
        """
        self._guard_mutable("변환 실패 기록")
        if not reason.strip():
            raise InvariantViolation(
                "실패 이유가 없으면 다음 사람이 또 시도한다.", subject="reason"
            )
        self._rejections.append((label, reason.strip()))
        self._record(
            domain_events.ConversionRejected(
                run_id=self._id, label=label, reason=reason.strip()
            )
        )

    def select(self, policy: SelectionPolicy) -> OptimizationCertificate:
        """예산 안에서 고른다. (실습 4-10)"""
        if self._baseline_candidate is None:
            raise IllegalStateTransition(
                "기준을 재기 전에는 선택할 수 없다.", subject=str(self._id)
            )
        if not self._candidates:
            raise NoCandidateSelected(
                "후보가 하나도 없다. 기준 모델만으로는 최적화라고 할 수 없다.",
                subject=str(self._id),
            )

        certificate = policy.evaluate(self.tradeoff_table())
        self._certificate = certificate
        if certificate.has_selection:
            self._status = OptimizationStatus.SELECTED
            self._record(
                domain_events.ModelSelected(
                    run_id=self._id,
                    artifact_id=certificate.selected_artifact_id,
                    label=certificate.selected_label or "",
                    verdict=certificate.verdict,
                )
            )
        else:
            self._status = OptimizationStatus.BLOCKED
            self._record(
                domain_events.SelectionBlocked(
                    run_id=self._id,
                    reasons=tuple(
                        f"{v.label}: {'; '.join(v.reasons)}"
                        for v in certificate.verdicts
                        if not v.accepted
                    ),
                )
            )
        return certificate

    def reopen(self, reason: str) -> None:
        if self._status not in (
            OptimizationStatus.SELECTED,
            OptimizationStatus.BLOCKED,
        ):
            raise IllegalStateTransition(
                "판정되지 않은 최적화는 reopen 대상이 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "판정을 되돌리려면 이유를 남겨야 한다.", subject="reason"
            )
        self._status = OptimizationStatus.EXPLORING
        self._certificate = None
        self._record(
            domain_events.OptimizationRunReopened(
                run_id=self._id, reason=reason.strip()
            )
        )

    # -- 내부 --------------------------------------------------------------
    def _guard_mutable(self, action: str) -> None:
        if self._status in (OptimizationStatus.SELECTED, OptimizationStatus.BLOCKED):
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 '{action}' 을 할 수 없다. "
                "reopen(reason) 으로 판정을 되돌린 뒤 수정한다.",
                subject=str(self._id),
            )

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"OptimizationRun(id={self._id}, status={self._status.value}, "
            f"candidates={len(self._candidates)})"
        )
