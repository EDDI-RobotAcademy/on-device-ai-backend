"""ProfileDataset — 데이터를 실제로 열어 사실을 확보한다. (실습 1-1)

"데이터를 믿지 마라"가 코드가 되는 첫 지점이다.
여기서 나오는 숫자는 판단이 아니라 사실이다. 판단은 그 뒤에 온다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.data.dto import DatasetProfileView
from application.data.support import commit, load_dataset
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetProfiler, DatasetRepository


@dataclass(frozen=True, slots=True)
class ProfileDatasetCommand:
    dataset_id: str


class ProfileDataset:
    def __init__(
        self,
        repository: DatasetRepository,
        profiler: DatasetProfiler,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._profiler = profiler
        self._publisher = publisher

    def execute(self, command: ProfileDatasetCommand) -> DatasetProfileView:
        dataset = load_dataset(self._repository, command.dataset_id)

        # 측정은 Infrastructure 가 한다. Domain 은 결과만 받는다.
        profile = self._profiler.profile(dataset.source)

        dataset.attach_profile(profile)
        commit(self._repository, dataset, self._publisher)
        return DatasetProfileView.of(str(dataset.id), profile)
