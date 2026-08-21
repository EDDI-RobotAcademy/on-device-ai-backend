"""Model Use Case 결과 DTO."""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from domain.model.acceptance import ModelCertificate
from domain.model.architecture import ArchitectureProfile
from domain.model.curve import TrainingCurve
from domain.model.evaluation import EvaluationResult
from domain.model.experiment import ExperimentBoard
from domain.model.statistical_baseline import BaselineComparison
from domain.model.tensor_spec import DatasetTensorSummary
from domain.model.training_run import TrainingRun


@dataclass(frozen=True, slots=True)
class TensorSummaryView:
    split: str
    sample_count: int
    sample_shape: tuple[int, ...]
    class_counts: dict[str, int]
    feature_min: float | None
    feature_max: float | None
    feature_mean: float | None
    feature_std: float | None
    nan_count: int

    @classmethod
    def of(cls, summary: DatasetTensorSummary) -> TensorSummaryView:
        return cls(
            split=summary.split,
            sample_count=summary.sample_count,
            sample_shape=tuple(summary.sample_shape),
            class_counts=dict(summary.class_counts),
            feature_min=summary.feature_min,
            feature_max=summary.feature_max,
            feature_mean=summary.feature_mean,
            feature_std=summary.feature_std,
            nan_count=summary.nan_count,
        )

    def render(self) -> str:
        stats = (
            f"min={self.feature_min:>7.3f} max={self.feature_max:>7.3f} "
            f"mean={self.feature_mean:>7.3f} std={self.feature_std:>6.3f}"
            if self.feature_mean is not None
            else ""
        )
        classes = "  ".join(
            f"{name} {count}" for name, count in sorted(self.class_counts.items())
        )
        return (
            f"  {self.split:<11}{self.sample_count:>6,}개  "
            f"shape={str(self.sample_shape):<10}{stats}\n"
            f"              {classes}"
        )


@dataclass(frozen=True, slots=True)
class PreparationView:
    run_id: str
    dataset_ref: str
    architecture: str
    windowing: str
    input_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    bytes_per_batch: int
    summaries: tuple[TensorSummaryView, ...]
    windowing_report: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    def render(self) -> str:
        lines = [
            f"학습 준비 ({self.run_id})",
            f"  데이터   : {self.dataset_ref}",
            f"  구조     : {self.architecture}",
            f"  창       : {self.windowing}",
            f"  한 표본  : {self.input_shape}",
            f"  한 배치  : {self.batch_shape}  ({self.bytes_per_batch / 1024:.1f} KiB)",
            "",
            self.windowing_report,
            "",
        ]
        lines += [view.render() for view in self.summaries]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ArchitectureProfileView:
    run_id: str
    parameter_count: int
    mac_count: int
    parameter_bytes: int
    heaviest_layer: str | None
    busiest_layer: str | None
    table: str

    @classmethod
    def of(cls, run_id: str, profile: ArchitectureProfile) -> ArchitectureProfileView:
        heaviest = profile.heaviest_layer
        busiest = profile.busiest_layer
        return cls(
            run_id=run_id,
            parameter_count=profile.parameter_count,
            mac_count=profile.mac_count,
            parameter_bytes=profile.parameter_bytes,
            heaviest_layer=heaviest.name if heaviest else None,
            busiest_layer=busiest.name if busiest else None,
            table=profile.render(),
        )

    def render(self) -> str:
        return (
            f"{self.table}\n"
            f"  가장 무거운 층: {self.heaviest_layer} / "
            f"가장 바쁜 층: {self.busiest_layer}"
        )


@dataclass(frozen=True, slots=True)
class TrainingCurveView:
    run_id: str
    status: str
    epoch_count: int
    best_epoch: int | None
    train_loss_drop: float
    overfitting_epoch: int | None
    final_gap: float
    wasted_epochs: int
    total_seconds: float
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        status: str,
        curve: TrainingCurve,
        findings: tuple[FindingView, ...] = (),
    ) -> TrainingCurveView:
        best = curve.best_epoch
        return cls(
            run_id=run_id,
            status=status,
            epoch_count=len(curve),
            best_epoch=best.epoch if best else None,
            train_loss_drop=curve.train_loss_drop,
            overfitting_epoch=curve.overfitting_epoch,
            final_gap=curve.final_gap,
            wasted_epochs=curve.wasted_epochs,
            total_seconds=curve.total_seconds,
            table=curve.render(),
            findings=findings,
        )

    def render(self) -> str:
        lines = [
            f"학습 곡선 ({self.run_id}) — {self.status}",
            self.table,
            "",
            f"  학습 손실 감소 {self.train_loss_drop:.1%} / "
            f"최저점 epoch {self.best_epoch} / "
            f"일반화 격차 {self.final_gap:+.3f}",
        ]
        if self.overfitting_epoch:
            lines.append(f"  외우기 시작한 지점: epoch {self.overfitting_epoch}")
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class EvaluationView:
    run_id: str
    split: str
    accuracy: float
    baseline_accuracy: float
    macro_recall: float
    macro_f1: float
    loss: float
    latency_ms_p50: float
    latency_ms_p95: float
    per_class: tuple[tuple[str, int, float, float, float], ...]
    never_predicted: tuple[str, ...]
    matrix: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        result: EvaluationResult,
        findings: tuple[FindingView, ...] = (),
    ) -> EvaluationView:
        matrix = result.matrix
        return cls(
            run_id=run_id,
            split=result.split,
            accuracy=matrix.accuracy,
            baseline_accuracy=matrix.baseline_accuracy,
            macro_recall=matrix.macro_recall,
            macro_f1=matrix.macro_f1,
            loss=result.loss,
            latency_ms_p50=result.latency_ms_p50,
            latency_ms_p95=result.latency_ms_p95,
            per_class=tuple(
                (
                    label,
                    matrix.support_of(label),
                    matrix.recall_of(label),
                    matrix.precision_of(label),
                    matrix.f1_of(label),
                )
                for label in matrix.labels
            ),
            never_predicted=matrix.never_predicted,
            matrix=matrix.render(),
            findings=findings,
        )

    def recall_of(self, label: str) -> float:
        for name, _, recall, _, _ in self.per_class:
            if name == label:
                return recall
        return 0.0

    def render(self) -> str:
        lines = [
            f"평가 ({self.run_id}) — {self.split}",
            self.matrix,
            "",
            f"  p50 {self.latency_ms_p50:.3f}ms / p95 {self.latency_ms_p95:.3f}ms",
        ]
        if self.findings:
            lines.append("")
            lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ModelCertificateView:
    run_id: str
    model_version_id: str
    verdict: str
    is_deployable: bool
    accuracy: float
    macro_recall: float
    latency_ms_p95: float
    blocking: tuple[FindingView, ...]
    warnings: tuple[FindingView, ...]

    @classmethod
    def of(cls, run_id: str, certificate: ModelCertificate) -> ModelCertificateView:
        return cls(
            run_id=run_id,
            model_version_id=str(certificate.model_version_id),
            verdict=certificate.verdict.value,
            is_deployable=certificate.is_deployable,
            accuracy=certificate.accuracy,
            macro_recall=certificate.macro_recall,
            latency_ms_p95=certificate.latency_ms_p95,
            blocking=tuple(FindingView.of(f) for f in certificate.blocking),
            warnings=tuple(FindingView.of(f) for f in certificate.warnings),
        )

    def render(self) -> str:
        lines = [
            f"모델 승인 판정: {self.verdict}  ({self.model_version_id})",
            f"  정확도 {self.accuracy:.3f} / macro recall {self.macro_recall:.3f} "
            f"/ p95 {self.latency_ms_p95:.2f}ms",
        ]
        if self.blocking:
            lines.append("")
            lines.append("차단 사유:")
            lines += [f"  ✗ {f.describe()}" for f in self.blocking]
        if self.warnings:
            lines.append("")
            lines.append("경고(배포는 가능):")
            lines += [f"  ! {f.describe()}" for f in self.warnings]
        if self.is_deployable and not self.warnings:
            lines.append("")
            lines.append("  ✓ 막는 것도, 걸리는 것도 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TrainingRunView:
    run_id: str
    dataset_ref: str
    status: str
    architecture: str
    config: str
    windowing: str
    epoch_count: int
    best_epoch: int | None
    model_version_id: str | None
    verdict: str | None
    failure_reason: str | None
    evaluated_splits: tuple[str, ...]

    @classmethod
    def of(cls, run: TrainingRun) -> TrainingRunView:
        best = run.curve.best_epoch
        return cls(
            run_id=str(run.id),
            dataset_ref=run.data.dataset_ref,
            status=run.status.value,
            architecture=run.architecture.describe(),
            config=run.config.describe(),
            # 이미지 학습에는 창이 없다 (실습 3-11).
            windowing=(
                run.windowing.describe()
                if run.windowing is not None
                else "창 없음 — 이미지에는 자를 시간 축이 없다"
            ),
            epoch_count=len(run.curve),
            best_epoch=best.epoch if best else None,
            model_version_id=(
                str(run.model_version_id) if run.model_version_id else None
            ),
            verdict=(
                run.certificate.verdict.value if run.certificate else None
            ),
            failure_reason=run.failure_reason,
            evaluated_splits=tuple(
                sorted(
                    split
                    for split in ("train", "validation", "test", "field")
                    if run.evaluation_of(split) is not None
                )
            ),
        )


# ---------------------------------------------------------------------------
# 실험 비교 (실습 3-12, 3-14, 3-15, 6-12)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExperimentTrialView:
    label: str
    knobs: str
    accuracy: float
    macro_recall: float
    macro_f1: float
    loss: float
    latency_ms_p50: float
    parameter_count: int
    epochs: int
    evaluated_samples: int
    seed: int
    data_ref: str


@dataclass(frozen=True, slots=True)
class ExperimentBoardView:
    """실험 비교표. HTTP 로 그대로 나가는 모양이다."""

    name: str
    metric: str
    trials: tuple[ExperimentTrialView, ...]
    best_label: str | None
    gap_to_runner_up: float
    spread: float
    table: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        board: ExperimentBoard,
        *,
        metric: str = "macro_f1",
        findings: tuple[FindingView, ...] = (),
    ) -> ExperimentBoardView:
        best = board.best_by(metric) if not board.is_empty else None
        return cls(
            name=board.name,
            metric=metric,
            trials=tuple(
                ExperimentTrialView(
                    label=t.label,
                    knobs=t.knobs.describe(),
                    accuracy=t.metrics.accuracy,
                    macro_recall=t.metrics.macro_recall,
                    macro_f1=t.metrics.macro_f1,
                    loss=t.metrics.loss,
                    latency_ms_p50=t.metrics.latency_ms_p50,
                    parameter_count=t.metrics.parameter_count,
                    epochs=t.metrics.epochs,
                    evaluated_samples=t.metrics.evaluated_samples,
                    seed=t.seed,
                    data_ref=t.data_ref,
                )
                for t in board.trials
            ),
            best_label=best.label if best else None,
            gap_to_runner_up=board.gap_to_runner_up(metric),
            spread=board.spread_of(metric),
            table=board.render(metric),
            findings=findings,
        )

    def render(self) -> str:
        lines = [self.table]
        if self.findings:
            lines.append("")
            lines.extend(f"  - {f.describe()}" for f in self.findings)
        return "\n".join(lines)

    def trial_of(self, label: str) -> ExperimentTrialView | None:
        return next((t for t in self.trials if t.label == label), None)


@dataclass(frozen=True, slots=True)
class BaselineComparisonView:
    """통계 기준선 대 학습 모델. (실습 3-13)"""

    run_id: str
    detector: str
    statistical_recall: float
    statistical_precision: float
    model_recall: float
    model_precision: float
    recall_gain: float
    precision_gain: float
    model_type_accuracy: float
    type_count: int
    table: str
    statistical_matrix: str
    model_matrix: str
    findings: tuple[FindingView, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        run_id: str,
        comparison: BaselineComparison,
        *,
        statistical_matrix: str,
        model_matrix: str,
        findings: tuple[FindingView, ...] = (),
    ) -> BaselineComparisonView:
        return cls(
            run_id=run_id,
            detector=comparison.detector,
            statistical_recall=comparison.statistical_recall,
            statistical_precision=comparison.statistical_precision,
            model_recall=comparison.model_recall,
            model_precision=comparison.model_precision,
            recall_gain=comparison.recall_gain,
            precision_gain=comparison.precision_gain,
            model_type_accuracy=comparison.model_type_accuracy,
            type_count=comparison.type_count,
            table=comparison.render(),
            statistical_matrix=statistical_matrix,
            model_matrix=model_matrix,
            findings=findings,
        )

    def render(self) -> str:
        lines = [self.table]
        if self.findings:
            lines.append("")
            lines.extend(f"  - {f.describe()}" for f in self.findings)
        return "\n".join(lines)
