"""CompareCandidates — 정확도와 Latency를 한 표에 놓는다. (실습 4-5, 4-8)

이 Use Case 는 아무것도 바꾸지 않는다. **읽고 늘어놓기만 한다.**
그런데 이 표가 없으면 "빨라졌습니다"만 남고 무엇을 잃었는지는 아무도 모른다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.optimization.dto import CandidateView, TradeoffView
from application.optimization.support import load_run
from domain.optimization.ports import OptimizationRunRepository


@dataclass(frozen=True, slots=True)
class CompareCandidatesCommand:
    run_id: str


class CompareCandidates:
    def __init__(self, runs: OptimizationRunRepository) -> None:
        self._runs = runs

    def execute(self, command: CompareCandidatesCommand) -> TradeoffView:
        run = load_run(self._runs, command.run_id)
        return TradeoffView.of(str(run.id), run.tradeoff_table())


@dataclass(frozen=True, slots=True)
class InspectArtifactSizesCommand:
    """모델 크기가 왜 이론값과 다른가. (실습 4-5)"""

    run_id: str


class InspectArtifactSizes:
    """후보마다 '가중치 + 오버헤드'로 크기를 쪼개 보여준다.

    3,187개 파라미터 × 4바이트 = 12,748바이트.
    그런데 파일은 16KB 다. 나머지는 그래프 구조와 연산자 목록이다.
    작은 모델일수록 이 몫이 커서, 양자화 이득이 이론만큼 나오지 않는다.
    """

    def __init__(self, runs: OptimizationRunRepository) -> None:
        self._runs = runs

    def execute(
        self, command: InspectArtifactSizesCommand
    ) -> tuple[CandidateView, ...]:
        run = load_run(self._runs, command.run_id)
        table = run.tradeoff_table()
        return tuple(
            CandidateView.of(str(run.id), candidate)
            for candidate in table.all_candidates
        )
