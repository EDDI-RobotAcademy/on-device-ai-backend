"""CertifyDatasetReadiness — 학습을 시작해도 되는지 판정한다. (실습 1-10)

이 Use Case 가 하는 일은 놀랄 만큼 적다.
    Dataset 에게 "네 상태로 판정해봐라"고 시키는 것이 전부다.
판정 규칙은 ReadinessPolicy 가, 상태는 Dataset 이 들고 있다.
Application 이 if 문으로 통과 여부를 계산하기 시작하면 그 순간 설계가 무너진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import ReadinessView
from application.data.support import commit, load_dataset
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository
from domain.data.readiness import ReadinessPolicy


@dataclass(frozen=True, slots=True)
class CertifyDatasetReadinessCommand:
    dataset_id: str
    policy: ReadinessPolicy = field(default_factory=ReadinessPolicy)


class CertifyDatasetReadiness:
    def __init__(
        self,
        repository: DatasetRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: CertifyDatasetReadinessCommand) -> ReadinessView:
        dataset = load_dataset(self._repository, command.dataset_id)
        certificate = dataset.certify(command.policy)
        commit(self._repository, dataset, self._publisher)
        return ReadinessView.of(certificate)


@dataclass(frozen=True, slots=True)
class ReopenDatasetCommand:
    dataset_id: str
    reason: str


class ReopenDataset:
    """판정을 되돌린다. 데이터를 고쳤으면 다시 판정받아야 한다."""

    def __init__(
        self,
        repository: DatasetRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher

    def execute(self, command: ReopenDatasetCommand) -> str:
        dataset = load_dataset(self._repository, command.dataset_id)
        dataset.reopen(command.reason)
        commit(self._repository, dataset, self._publisher)
        return dataset.status.value
