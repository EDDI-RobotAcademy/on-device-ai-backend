"""저장소·실험·엔드포인트 Use Case. (실습 6-12, 6-13, 6-14)

셋 다 모양이 같다.

    바깥에 무언가를 걸거나 남긴다 → 그 결과를 **읽는다** → Domain 이 판정한다

읽는 단계를 빼면 "설정했다"까지만 알고 "설정되어 있다"는 모른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.fleet.dto import (
    BucketGovernanceView,
    EndpointView,
    ExperimentLedgerView,
)
from domain.fleet.endpoint import EndpointPolicy, EndpointSpec, OnlineInferenceProfile
from domain.fleet.experiment_record import (
    ExperimentRecord,
    ReproducibilityPolicy,
)
from domain.fleet.governance import AccessStatement, GovernancePolicy
from domain.fleet.ports import (
    BucketGovernanceGateway,
    EndpointGateway,
    ExperimentStore,
)


# ---------------------------------------------------------------------------
# 6-13
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GovernStorageCommand:
    versioning: bool = True
    encryption: str | None = "AES256"
    block_public_access: bool = True
    expiration_days: int | None = 365
    statements: tuple[AccessStatement, ...] = ()
    version_prefix: str = ""
    policy: GovernancePolicy = field(default_factory=GovernancePolicy)


class GovernStorage:
    def __init__(self, gateway: BucketGovernanceGateway) -> None:
        self._gateway = gateway

    def execute(self, command: GovernStorageCommand) -> BucketGovernanceView:
        self._gateway.harden(
            versioning=command.versioning,
            encryption=command.encryption,
            block_public_access=command.block_public_access,
            expiration_days=command.expiration_days,
        )
        if command.statements:
            self._gateway.put_policy(command.statements)

        governance = self._gateway.inspect(version_prefix=command.version_prefix)
        return BucketGovernanceView.of(
            governance,
            findings=tuple(
                FindingView.of(f) for f in command.policy.inspect(governance)
            ),
        )


@dataclass(frozen=True, slots=True)
class InspectStorageCommand:
    version_prefix: str = ""
    policy: GovernancePolicy = field(default_factory=GovernancePolicy)


class InspectStorage:
    """**걸지 않고 읽기만 한다.** 기본값 그대로인 버킷을 보는 자리다."""

    def __init__(self, gateway: BucketGovernanceGateway) -> None:
        self._gateway = gateway

    def execute(self, command: InspectStorageCommand) -> BucketGovernanceView:
        governance = self._gateway.inspect(version_prefix=command.version_prefix)
        return BucketGovernanceView.of(
            governance,
            findings=tuple(
                FindingView.of(f) for f in command.policy.inspect(governance)
            ),
        )


# ---------------------------------------------------------------------------
# 6-12
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RecordExperimentCommand:
    records: tuple[ExperimentRecord, ...]


class RecordExperiment:
    def __init__(self, store: ExperimentStore) -> None:
        self._store = store

    def execute(self, command: RecordExperimentCommand) -> tuple[str, ...]:
        return tuple(self._store.record(entry) for entry in command.records)


@dataclass(frozen=True, slots=True)
class ReviewExperimentCommand:
    experiment_id: str
    metric: str = "macro_f1"
    policy: ReproducibilityPolicy = field(default_factory=ReproducibilityPolicy)


class ReviewExperiment:
    def __init__(self, store: ExperimentStore) -> None:
        self._store = store

    def execute(self, command: ReviewExperimentCommand) -> ExperimentLedgerView:
        ledger = self._store.load(command.experiment_id)
        return ExperimentLedgerView.of(
            ledger,
            metric=command.metric,
            findings=tuple(FindingView.of(f) for f in command.policy.inspect(ledger)),
        )


# ---------------------------------------------------------------------------
# 6-14
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DeployEndpointCommand:
    spec: EndpointSpec
    profile: OnlineInferenceProfile
    image_uri: str = "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/serve:1"
    policy: EndpointPolicy = field(default_factory=EndpointPolicy)


class DeployEndpoint:
    def __init__(self, gateway: EndpointGateway) -> None:
        self._gateway = gateway

    def execute(self, command: DeployEndpointCommand) -> EndpointView:
        state = self._gateway.deploy(command.spec, image_uri=command.image_uri)
        findings = command.policy.inspect(command.spec, command.profile)
        return EndpointView.of(
            command.spec,
            state,
            command.profile,
            findings=tuple(FindingView.of(f) for f in findings),
        )


@dataclass(frozen=True, slots=True)
class TeardownEndpointCommand:
    name: str


class TeardownEndpoint:
    """**실습이 끝나면 반드시 부른다.** 켜 두면 시간당 과금된다."""

    def __init__(self, gateway: EndpointGateway) -> None:
        self._gateway = gateway

    def execute(self, command: TeardownEndpointCommand) -> None:
        self._gateway.teardown(command.name)
