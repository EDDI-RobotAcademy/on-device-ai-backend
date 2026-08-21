"""TrainingRun — Model Context 의 Aggregate Root.

학습은 **오래 걸리는 작업**이다 (CLAUDE.md §11).
HTTP 요청이 그것을 붙잡고 기다리는 구조를 만들지 않는다.

    Command  →  Job          →  Job Status         →  Result
    학습 요청 →  TrainingRun  →  RUNNING/COMPLETED  →  ModelVersion

그래서 이 Aggregate 는 '학습하는 객체'가 아니라 **'학습이라는 사건의 기록'**이다.
계산은 Infrastructure 가 하고, 여기에는 그 진행과 결과가 남는다.

지키는 불변식:
    - 게이트를 통과하지 않은 데이터로는 시작조차 할 수 없다.
    - 시작하지 않은 학습에 epoch 을 기록할 수 없다.
    - epoch 은 순서대로만 기록된다.
    - 끝나지 않은 학습을 평가할 수 없다.
    - 승인된 모델은 몰래 바뀌지 않는다.
"""

from __future__ import annotations

from enum import Enum

from domain.model import events as domain_events
from domain.model.acceptance import ModelAcceptancePolicy, ModelCertificate
from domain.model.architecture import ArchitectureProfile, ModelArchitecture
from domain.model.curve import EpochRecord, TrainingCurve
from domain.model.errors import ModelNotTrained, ShapeMismatch
from domain.model.evaluation import EvaluationResult
from domain.model.identifiers import ModelVersionId, TrainingRunId
from domain.model.image_data_ref import ImageDataRef
from domain.model.protocol import SplitUsage
from domain.model.tensor_spec import DatasetTensorSummary
from domain.model.training_config import TrainingConfig
from domain.model.training_data_ref import TrainingDataRef
from domain.model.windowing import WindowingPlan, WindowingSummary
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.events import EventRecorder


class TrainingStatus(Enum):
    PREPARED = "PREPARED"
    """구조와 데이터가 정해졌다. 아직 돌지 않았다."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ACCEPTED = "ACCEPTED"
    """현장에 내보낼 수 있다고 판정했다."""

    REJECTED = "REJECTED"


class TrainingRun(EventRecorder):
    """한 번의 학습."""

    __slots__ = (
        "_id",
        "_data",
        "_architecture",
        "_config",
        "_windowing",
        "_windowing_summary",
        "_status",
        "_curve",
        "_profile",
        "_tensor_summaries",
        "_usage",
        "_evaluations",
        "_model_version_id",
        "_certificate",
        "_failure_reason",
    )

    def __init__(
        self,
        run_id: TrainingRunId,
        data: TrainingDataRef | ImageDataRef,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        windowing: WindowingPlan | None,
    ) -> None:
        super().__init__()
        self._id = run_id
        self._data = data
        self._architecture = architecture
        self._config = config
        self._windowing = windowing
        self._windowing_summary: WindowingSummary | None = None
        self._status = TrainingStatus.PREPARED
        self._curve = TrainingCurve()
        self._profile: ArchitectureProfile | None = None
        self._tensor_summaries: dict[str, DatasetTensorSummary] = {}
        self._usage: SplitUsage | None = None
        self._evaluations: dict[str, EvaluationResult] = {}
        self._model_version_id: ModelVersionId | None = None
        self._certificate: ModelCertificate | None = None
        self._failure_reason: str | None = None

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def prepare(
        cls,
        run_id: TrainingRunId,
        data: TrainingDataRef,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        windowing: WindowingPlan,
        *,
        require_gates: bool = True,
    ) -> TrainingRun:
        """학습을 준비한다.

        게이트를 통과하지 않은 데이터라면 **여기서 멈춘다.**
        모듈 1과 2가 존재한 이유가 이 한 줄이다.
        """
        if require_gates and not data.gates_passed:
            raise IllegalStateTransition(
                "게이트를 통과하지 않은 데이터로 학습을 시작할 수 없다: "
                + ", ".join(data.missing_gates),
                subject=str(run_id),
            )
        if architecture.input_spec.shape[-1] != data.feature_count and (
            len(architecture.input_spec.shape) > 1
        ):
            # (시간, 채널) 배치에서 마지막 축이 채널이다.
            raise ShapeMismatch(
                f"모델 입력의 채널 수({architecture.input_spec.shape[-1]})와 "
                f"데이터의 입력 필드 수({data.feature_count})가 다르다.",
                subject=str(run_id),
            )
        if architecture.input_spec.shape[0] != windowing.window_length:
            raise ShapeMismatch(
                f"창 길이({windowing.window_length})와 모델 입력의 시간 축"
                f"({architecture.input_spec.shape[0]})이 다르다. "
                "학습은 돌아가고 배포할 때 터진다.",
                subject=str(run_id),
            )

        run = cls(run_id, data, architecture, config, windowing)
        run._record(
            domain_events.TrainingRunPrepared(
                run_id=run_id,
                dataset_ref=data.dataset_ref,
                architecture=architecture.describe(),
            )
        )
        return run

    @classmethod
    def prepare_images(
        cls,
        run_id: TrainingRunId,
        data: ImageDataRef,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        *,
        require_gates: bool = True,
    ) -> TrainingRun:
        """이미지로 학습을 준비한다. (실습 3-11)

        표(CSV)와 같은 Aggregate 를 쓴다. **학습이라는 사건은 하나이기 때문이다.**
        달라지는 것은 재료를 어디서 가져오느냐뿐이다.

        그래서 창(window)이 없다. 이미지에는 자를 시간 축이 없다.
        """
        if require_gates and not data.gates_passed:
            raise IllegalStateTransition(
                "이미지 게이트를 통과하지 않은 데이터로 학습을 시작할 수 없다: "
                + ", ".join(data.missing_gates),
                subject=str(run_id),
            )
        if architecture.class_count != data.class_count:
            raise ShapeMismatch(
                f"모델의 출력 클래스 수({architecture.class_count})와 "
                f"폴더 수({data.class_count})가 다르다. "
                "폴더를 하나 더 만들면 모델도 다시 만들어야 한다.",
                subject=str(run_id),
            )
        expected = data.spec.to_tensor_spec().shape
        if tuple(architecture.input_spec.shape) != tuple(expected):
            raise ShapeMismatch(
                f"모델 입력 {architecture.input_spec.shape} 과 "
                f"이미지 명세 {expected} 가 다르다. "
                "전처리와 모델은 같은 계약을 봐야 한다.",
                subject=str(run_id),
            )

        run = cls(run_id, data, architecture, config, None)
        run._record(
            domain_events.TrainingRunPrepared(
                run_id=run_id,
                dataset_ref=data.dataset_ref,
                architecture=architecture.describe(),
            )
        )
        return run

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> TrainingRunId:
        return self._id

    @property
    def data(self) -> TrainingDataRef | ImageDataRef:
        return self._data

    @property
    def image_data(self) -> ImageDataRef:
        """이미지 학습일 때의 데이터 참조. 아니면 막는다."""
        if not isinstance(self._data, ImageDataRef):
            raise IllegalStateTransition(
                "이 학습은 이미지 학습이 아니다.", subject=str(self._id)
            )
        return self._data

    @property
    def is_image_run(self) -> bool:
        return isinstance(self._data, ImageDataRef)

    @property
    def architecture(self) -> ModelArchitecture:
        return self._architecture

    @property
    def config(self) -> TrainingConfig:
        return self._config

    @property
    def windowing(self) -> WindowingPlan | None:
        """이미지 학습에는 창이 없다. 그때는 None 이다."""
        return self._windowing

    @property
    def windowing_summary(self) -> WindowingSummary | None:
        return self._windowing_summary

    @property
    def status(self) -> TrainingStatus:
        return self._status

    @property
    def curve(self) -> TrainingCurve:
        return self._curve

    @property
    def profile(self) -> ArchitectureProfile | None:
        return self._profile

    @property
    def usage(self) -> SplitUsage | None:
        return self._usage

    @property
    def model_version_id(self) -> ModelVersionId | None:
        return self._model_version_id

    @property
    def certificate(self) -> ModelCertificate | None:
        return self._certificate

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def tensor_summaries(self) -> dict[str, DatasetTensorSummary]:
        return dict(self._tensor_summaries)

    def evaluation_of(self, split: str) -> EvaluationResult | None:
        return self._evaluations.get(split)

    @property
    def is_finished(self) -> bool:
        return self._status in (
            TrainingStatus.COMPLETED,
            TrainingStatus.ACCEPTED,
            TrainingStatus.REJECTED,
        )

    # -- 준비 단계 ---------------------------------------------------------
    def attach_tensor_summary(self, summary: DatasetTensorSummary) -> None:
        """데이터를 텐서로 바꿔 본 결과를 붙인다. (실습 3-1)"""
        self._guard_mutable("텐서 요약 기록")
        if not summary.matches(self._architecture.input_spec):
            raise ShapeMismatch(
                f"'{summary.split}' 의 표본 모양 {summary.sample_shape} 이 "
                f"모델 입력 {self._architecture.input_spec.shape} 과 다르다.",
                subject=summary.split,
            )
        self._tensor_summaries[summary.split] = summary

    def attach_windowing_summary(self, summary: WindowingSummary) -> None:
        """창으로 자른 결과를 붙인다. (실습 3-4)"""
        self._guard_mutable("창 요약 기록")
        if self._windowing is None:
            raise IllegalStateTransition(
                "창 계획이 없는 학습(이미지)에 창 요약을 붙일 수 없다.",
                subject=str(self._id),
            )
        if summary.window_length != self._windowing.window_length:
            raise ShapeMismatch(
                "요약의 창 길이가 계획과 다르다.", subject="window_length"
            )
        self._windowing_summary = summary

    def attach_architecture_profile(self, profile: ArchitectureProfile) -> None:
        """구조를 조립해 본 결과를 붙인다. (실습 3-2)"""
        self._guard_mutable("구조 프로파일 기록")
        self._profile = profile

    def attach_split_usage(self, usage: SplitUsage) -> None:
        """분할을 어떻게 썼는지 기록한다. (실습 3-8)"""
        self._guard_mutable("분할 사용 기록")
        self._usage = usage

    # -- 학습 --------------------------------------------------------------
    def start(self) -> None:
        if self._status is not TrainingStatus.PREPARED:
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 학습을 시작할 수 없다.",
                subject=str(self._id),
            )
        self._status = TrainingStatus.RUNNING
        self._record(
            domain_events.TrainingRunStarted(
                run_id=self._id, config=self._config.describe()
            )
        )

    def record_epoch(self, record: EpochRecord) -> None:
        if self._status is not TrainingStatus.RUNNING:
            raise IllegalStateTransition(
                "시작하지 않은 학습에 epoch 을 기록할 수 없다.", subject=str(self._id)
            )
        last = self._curve.last
        if last is not None and record.epoch <= last.epoch:
            raise InvariantViolation(
                f"epoch 은 순서대로 기록되어야 한다. "
                f"(마지막 {last.epoch}, 받은 값 {record.epoch})",
                subject="epoch",
            )
        self._curve = TrainingCurve(records=(*self._curve.records, record))
        self._record(
            domain_events.EpochCompleted(
                run_id=self._id,
                epoch=record.epoch,
                train_loss=record.train_loss,
                validation_loss=record.validation_loss,
            )
        )

    def complete(self, model_version_id: ModelVersionId) -> None:
        if self._status is not TrainingStatus.RUNNING:
            raise IllegalStateTransition(
                "돌고 있지 않은 학습을 끝낼 수 없다.", subject=str(self._id)
            )
        if self._curve.is_empty:
            raise InvariantViolation(
                "epoch 이 하나도 없는 학습을 완료로 볼 수 없다.", subject=str(self._id)
            )
        self._status = TrainingStatus.COMPLETED
        self._model_version_id = model_version_id
        best = self._curve.best_epoch
        self._record(
            domain_events.TrainingRunCompleted(
                run_id=self._id,
                model_version_id=model_version_id,
                epochs=len(self._curve),
                best_epoch=best.epoch if best else 0,
            )
        )

    def fail(self, reason: str) -> None:
        if self._status not in (TrainingStatus.PREPARED, TrainingStatus.RUNNING):
            raise IllegalStateTransition(
                "이미 끝난 학습을 실패로 바꿀 수 없다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "실패 이유가 없으면 다시 시도할 근거가 없다.", subject="reason"
            )
        self._status = TrainingStatus.FAILED
        self._failure_reason = reason.strip()
        self._record(
            domain_events.TrainingRunFailed(run_id=self._id, reason=reason.strip())
        )

    # -- 평가와 판정 -------------------------------------------------------
    def record_evaluation(self, result: EvaluationResult) -> None:
        """평가 결과를 붙인다. (실습 3-9)"""
        if self._status not in (
            TrainingStatus.COMPLETED,
            TrainingStatus.ACCEPTED,
            TrainingStatus.REJECTED,
        ):
            raise ModelNotTrained(
                "끝나지 않은 학습을 평가할 수 없다.", subject=str(self._id)
            )
        if self._status in (TrainingStatus.ACCEPTED, TrainingStatus.REJECTED):
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 평가를 다시 기록할 수 없다. "
                "reopen(reason) 으로 판정을 되돌린 뒤 수정한다.",
                subject=str(self._id),
            )
        self._evaluations[result.split] = result
        self._record(
            domain_events.ModelEvaluated(
                run_id=self._id,
                split=result.split,
                accuracy=result.accuracy,
                macro_recall=result.macro_recall,
            )
        )

    def accept(
        self, policy: ModelAcceptancePolicy, *, split: str = "test"
    ) -> ModelCertificate:
        """현장에 내보낼 수 있는지 판정한다. (실습 3-10)"""
        if self._status is not TrainingStatus.COMPLETED:
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 승인 판정을 할 수 없다.",
                subject=str(self._id),
            )
        result = self._evaluations.get(split)
        if result is None:
            raise ModelNotTrained(
                f"'{split}' 분할에 대한 평가가 없다. 판정할 근거가 없다.",
                subject=str(self._id),
            )
        if self._usage is None:
            raise ModelNotTrained(
                "분할 사용 기록이 없다. 평가 규율(실습 3-8)을 확인할 수 없다.",
                subject=str(self._id),
            )
        assert self._model_version_id is not None  # complete() 가 보장한다

        certificate = policy.evaluate(
            self._model_version_id,
            curve=self._curve,
            result=result,
            usage=self._usage,
            baseline_accuracy=result.matrix.baseline_accuracy,
            gates_passed=self._data.gates_passed,
            missing_gates=self._data.missing_gates,
        )
        self._certificate = certificate
        if certificate.is_deployable:
            self._status = TrainingStatus.ACCEPTED
            self._record(
                domain_events.ModelAccepted(
                    run_id=self._id,
                    model_version_id=self._model_version_id,
                    verdict=certificate.verdict,
                    warning_count=len(certificate.warnings),
                )
            )
        else:
            self._status = TrainingStatus.REJECTED
            self._record(
                domain_events.ModelRejected(
                    run_id=self._id,
                    model_version_id=self._model_version_id,
                    reasons=tuple(f.describe() for f in certificate.blocking),
                )
            )
        return certificate

    def reopen(self, reason: str) -> None:
        if self._status not in (TrainingStatus.ACCEPTED, TrainingStatus.REJECTED):
            raise IllegalStateTransition(
                "판정되지 않은 학습은 reopen 대상이 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "판정을 되돌리려면 이유를 남겨야 한다.", subject="reason"
            )
        self._status = TrainingStatus.COMPLETED
        self._certificate = None

    # -- 내부 --------------------------------------------------------------
    def _guard_mutable(self, action: str) -> None:
        if self._status not in (TrainingStatus.PREPARED, TrainingStatus.RUNNING):
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 '{action}' 을 할 수 없다.",
                subject=str(self._id),
            )

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"TrainingRun(id={self._id}, status={self._status.value}, "
            f"epochs={len(self._curve)})"
        )
