"""Dataset — Data Context 의 Aggregate Root.

이 객체가 지키는 것은 데이터가 아니라 **순서와 계약**이다.

    등록 → 프로파일 → 스키마 선언 → 검사들 → 판정(READY)

지키는 불변식:
    - 열어보지도 않은 데이터에 스키마를 선언할 수 없다.
    - 스키마 없이 시간축/라벨/분할을 논할 수 없다.
    - 선언한 스키마에 없는 필드를 라벨이나 그룹으로 지목할 수 없다.
    - READY 판정을 받은 Dataset 은 몰래 바뀌지 않는다. 바꾸려면 reopen 해야 한다.

절대 하지 않는 것:
    - 파일을 읽지 않는다.
    - pandas 를 부르지 않는다.
    - 지금 몇 시인지 묻지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.data import events as domain_events
from domain.data.errors import SchemaMismatch
from domain.data.identifiers import DatasetId
from domain.data.inspection import InspectionKind, InspectionReport, Verdict
from domain.data.labeling import LabelSpace
from domain.data.partition import PartitionMeasurement, PartitionPlan
from domain.data.profile import DatasetProfile
from domain.data.readiness import ReadinessCertificate, ReadinessPolicy
from domain.data.schema import DataSchema, FieldRole
from domain.data.source import DataSourceDescriptor
from domain.data.training_spec import TrainingDataSpec
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.events import EventRecorder


class DatasetStatus(Enum):
    REGISTERED = "REGISTERED"
    """존재는 안다. 아직 아무것도 열어보지 않았다."""

    PROFILED = "PROFILED"
    """열어봤다. 무엇이 들어 있는지는 안다. 의미는 아직 모른다."""

    DECLARED = "DECLARED"
    """스키마를 선언했다. 이제 계약이 있다."""

    INSPECTED = "INSPECTED"
    """검사를 진행 중이다."""

    READY = "READY"
    """학습에 써도 된다고 판정했다."""

    REJECTED = "REJECTED"
    """학습에 쓸 수 없다고 판정했다."""


@dataclass(frozen=True, slots=True)
class PartitionOutcome:
    """계획과 실제 결과를 한 쌍으로 보관한다."""

    plan: PartitionPlan
    measurement: PartitionMeasurement


class Dataset(EventRecorder):
    """현장에서 받은 데이터 한 덩어리."""

    __slots__ = (
        "_id",
        "_name",
        "_source",
        "_status",
        "_profile",
        "_schema",
        "_label_space",
        "_training_spec",
        "_partition",
        "_reports",
        "_certificate",
    )

    def __init__(
        self,
        dataset_id: DatasetId,
        name: str,
        source: DataSourceDescriptor,
    ) -> None:
        super().__init__()
        if not name.strip():
            raise InvariantViolation("Dataset 이름은 비어 있을 수 없다.", subject="name")
        self._id = dataset_id
        self._name = name.strip()
        self._source = source
        self._status = DatasetStatus.REGISTERED
        self._profile: DatasetProfile | None = None
        self._schema: DataSchema | None = None
        self._label_space: LabelSpace | None = None
        self._training_spec: TrainingDataSpec | None = None
        self._partition: PartitionOutcome | None = None
        self._reports: dict[InspectionKind, InspectionReport] = {}
        self._certificate: ReadinessCertificate | None = None

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def register(
        cls, dataset_id: DatasetId, name: str, source: DataSourceDescriptor
    ) -> Dataset:
        dataset = cls(dataset_id, name, source)
        dataset._record(
            domain_events.DatasetRegistered(
                dataset_id=dataset_id,
                name=dataset._name,
                collected_from=source.collected_from,
            )
        )
        return dataset

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> DatasetId:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> DataSourceDescriptor:
        return self._source

    @property
    def status(self) -> DatasetStatus:
        return self._status

    @property
    def profile(self) -> DatasetProfile | None:
        return self._profile

    @property
    def schema(self) -> DataSchema | None:
        return self._schema

    @property
    def label_space(self) -> LabelSpace | None:
        return self._label_space

    @property
    def training_spec(self) -> TrainingDataSpec | None:
        return self._training_spec

    @property
    def partition(self) -> PartitionOutcome | None:
        return self._partition

    @property
    def certificate(self) -> ReadinessCertificate | None:
        return self._certificate

    @property
    def reports(self) -> dict[InspectionKind, InspectionReport]:
        return dict(self._reports)

    def report_of(self, kind: InspectionKind) -> InspectionReport | None:
        return self._reports.get(kind)

    @property
    def is_ready(self) -> bool:
        return self._status is DatasetStatus.READY

    # -- 행위 --------------------------------------------------------------
    def attach_profile(self, profile: DatasetProfile) -> None:
        """데이터를 열어본 결과를 붙인다. (실습 1-1)"""
        self._guard_mutable("프로파일 갱신")
        if profile.row_count == 0:
            raise InvariantViolation(
                "행이 0개다. 파일은 있는데 데이터가 없다.", subject=str(self._id)
            )
        self._profile = profile
        if self._status is DatasetStatus.REGISTERED:
            self._status = DatasetStatus.PROFILED
        self._record(
            domain_events.DatasetProfiled(
                dataset_id=self._id,
                row_count=profile.row_count,
                column_count=len(profile.columns),
            )
        )

    def declare_schema(self, schema: DataSchema) -> InspectionReport:
        """스키마를 확정하고 즉시 현실과 맞대어 본다. (실습 1-2, 1-3)

        선언 자체는 막지 않는다. 불일치는 CRITICAL Finding 으로 남고,
        학습 착수 판정(certify)에서 차단된다. — 문제를 숨기지 않고 드러낸다.
        """
        self._guard_mutable("스키마 선언")
        if self._profile is None:
            raise IllegalStateTransition(
                "열어보지도 않은 데이터에 스키마를 선언할 수 없다. 먼저 프로파일링한다.",
                subject=str(self._id),
            )
        report = schema.inspect(self._profile)
        self._schema = schema
        if self._status is DatasetStatus.PROFILED:
            self._status = DatasetStatus.DECLARED
        self._record(
            domain_events.DataSchemaDeclared(
                dataset_id=self._id, field_count=len(schema.fields)
            )
        )
        self.record_inspection(report)
        return report

    def record_inspection(self, report: InspectionReport) -> None:
        """검사 결과를 붙인다. 같은 종류를 다시 검사하면 덮어쓴다(재검사)."""
        self._guard_mutable(f"{report.kind.value} 검사 기록")
        if report.kind is not InspectionKind.SCHEMA and self._schema is None:
            raise IllegalStateTransition(
                f"스키마 없이 {report.kind.value} 검사를 기록할 수 없다.",
                subject=str(self._id),
            )
        self._reports[report.kind] = report
        if self._status in (DatasetStatus.DECLARED, DatasetStatus.PROFILED):
            self._status = DatasetStatus.INSPECTED
        self._record(
            domain_events.InspectionRecorded(
                dataset_id=self._id,
                kind=report.kind,
                verdict=report.verdict,
                finding_count=len(report.findings),
            )
        )

    def define_label_space(self, space: LabelSpace) -> None:
        """정상/이상의 정의를 확정한다. (실습 1-6)"""
        self._guard_mutable("라벨 정의")
        schema = self._require_schema("라벨 정의")
        spec = schema.field_of(space.field_name)  # 없으면 UnknownField
        if spec.role is not FieldRole.LABEL:
            raise SchemaMismatch(
                f"'{space.field_name}' 의 역할은 {spec.role.value} 다. LABEL 이 아니다.",
                subject=space.field_name,
            )
        self._label_space = space
        self._record(
            domain_events.LabelSpaceDefined(
                dataset_id=self._id, class_count=len(space.definitions)
            )
        )

    def design_training_data(self, spec: TrainingDataSpec) -> InspectionReport:
        """모델 입력 계약을 확정한다. (실습 1-7)"""
        self._guard_mutable("학습 데이터 설계")
        schema = self._require_schema("학습 데이터 설계")
        if spec.schema != schema:
            raise SchemaMismatch(
                "학습 설계가 참조하는 스키마가 이 Dataset 의 스키마와 다르다.",
                subject=str(self._id),
            )
        if self._label_space is None:
            raise IllegalStateTransition(
                "정상/이상의 정의 없이 학습 데이터를 설계할 수 없다. 먼저 LabelSpace 를 정의한다.",
                subject=str(self._id),
            )
        if spec.label_field != self._label_space.field_name:
            raise SchemaMismatch(
                f"학습 설계의 라벨 필드('{spec.label_field}')가 "
                f"정의된 라벨 필드('{self._label_space.field_name}')와 다르다.",
                subject=spec.label_field,
            )
        self._training_spec = spec
        report = spec.inspect()
        self._record(
            domain_events.TrainingDataDesigned(
                dataset_id=self._id, input_shape=spec.input_shape
            )
        )
        self.record_inspection(report)
        return report

    def apply_partition(
        self,
        plan: PartitionPlan,
        measurement: PartitionMeasurement,
        report: InspectionReport,
    ) -> None:
        """분할 계획과 그 실측 결과를 확정한다. (실습 1-8)"""
        self._guard_mutable("분할 적용")
        schema = self._require_schema("분할 적용")
        plan.validate_against(schema)
        self._partition = PartitionOutcome(plan=plan, measurement=measurement)
        self._record(
            domain_events.DatasetPartitioned(
                dataset_id=self._id,
                strategy=plan.strategy.value,
                train_count=measurement.train_count,
                validation_count=measurement.validation_count,
                test_count=measurement.test_count,
            )
        )
        self.record_inspection(report)

    def certify(self, policy: ReadinessPolicy) -> ReadinessCertificate:
        """학습을 시작해도 되는지 판정한다. (실습 1-10)"""
        if self._status in (DatasetStatus.REGISTERED, DatasetStatus.PROFILED):
            raise IllegalStateTransition(
                "스키마조차 선언되지 않은 Dataset 을 판정할 수 없다.", subject=str(self._id)
            )
        certificate = policy.evaluate(self._id, self._reports)
        self._certificate = certificate
        if certificate.is_ready:
            self._status = DatasetStatus.READY
            self._record(
                domain_events.DatasetCertifiedReady(
                    dataset_id=self._id,
                    verdict=certificate.verdict,
                    warning_count=len(certificate.warning_findings),
                )
            )
        else:
            self._status = DatasetStatus.REJECTED
            self._record(
                domain_events.DatasetRejected(
                    dataset_id=self._id,
                    blocking_count=len(certificate.blocking_findings),
                    reasons=certificate.reasons(),
                )
            )
        return certificate

    def reopen(self, reason: str) -> None:
        """판정을 되돌리고 다시 손볼 수 있게 한다.

        READY 인 Dataset 을 조용히 바꾸지 못하게 하는 것이 이 메서드의 존재 이유다.
        """
        if self._status not in (DatasetStatus.READY, DatasetStatus.REJECTED):
            raise IllegalStateTransition(
                "판정되지 않은 Dataset 은 reopen 대상이 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "판정을 되돌리려면 이유를 남겨야 한다.", subject="reason"
            )
        self._status = DatasetStatus.INSPECTED
        self._certificate = None
        self._record(
            domain_events.DatasetReopened(dataset_id=self._id, reason=reason.strip())
        )

    # -- 내부 --------------------------------------------------------------
    def _guard_mutable(self, action: str) -> None:
        if self._status in (DatasetStatus.READY, DatasetStatus.REJECTED):
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 '{action}' 을 할 수 없다. "
                "reopen(reason) 으로 판정을 되돌린 뒤 수정한다.",
                subject=str(self._id),
            )

    def _require_schema(self, action: str) -> DataSchema:
        if self._schema is None:
            raise IllegalStateTransition(
                f"스키마 없이 '{action}' 을 할 수 없다.", subject=str(self._id)
            )
        return self._schema

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return f"Dataset(id={self._id}, name={self._name!r}, status={self._status.value})"

    @property
    def latest_verdict(self) -> Verdict | None:
        return self._certificate.verdict if self._certificate else None
