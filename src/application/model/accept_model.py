"""AcceptModel — 현장 데이터를 통과하는 모델만 살아남는다. (실습 3-10)

모듈 1의 CertifyDatasetReadiness, 모듈 2의 EvaluateQualityGate 와 같은 자리다.
묻는 것만 다르다.

    모듈 1  이 데이터가 무엇인지 아는가
    모듈 2  이 데이터가 쓸 만한가
    모듈 3  이 **모델**이 쓸 만한가
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.model.dto import ModelCertificateView
from application.model.support import commit, load_run
from application.shared.ports import EventPublisher
from domain.model.acceptance import ModelAcceptancePolicy
from domain.model.ports import TrainingRunRepository


@dataclass(frozen=True, slots=True)
class AcceptModelCommand:
    run_id: str
    split: str = "test"
    policy: ModelAcceptancePolicy = field(default_factory=ModelAcceptancePolicy)


class AcceptModel:
    def __init__(
        self,
        runs: TrainingRunRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._publisher = publisher

    def execute(self, command: AcceptModelCommand) -> ModelCertificateView:
        run = load_run(self._runs, command.run_id)
        certificate = run.accept(command.policy, split=command.split)
        commit(self._runs, run, self._publisher)
        return ModelCertificateView.of(str(run.id), certificate)


@dataclass(frozen=True, slots=True)
class ReopenTrainingRunCommand:
    run_id: str
    reason: str


class ReopenTrainingRun:
    def __init__(
        self,
        runs: TrainingRunRepository,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._publisher = publisher

    def execute(self, command: ReopenTrainingRunCommand) -> str:
        run = load_run(self._runs, command.run_id)
        run.reopen(command.reason)
        commit(self._runs, run, self._publisher)
        return run.status.value
