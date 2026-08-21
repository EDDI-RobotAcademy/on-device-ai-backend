"""FastAPI 애플리케이션 조립."""

from __future__ import annotations

from fastapi import FastAPI

from interfaces.http.errors import register_exception_handlers
from interfaces.http.routes import (
    datasets,
    extended,
    fleet,
    models,
    operations,
    optimization,
    quality,
)

DESCRIPTION = """
산업용 On-Device AI 시스템 Backend.

현재 구현 범위

**모듈 1 — 데이터** (`/datasets`)
    현장 데이터를 등록하고, 열어보고, 구조를 계약으로 만들고,
    신호·시간축·라벨·분할·대표성을 검증한 뒤,
    "이 데이터가 무엇인지 아는가"를 판정한다.

**모듈 2 — 데이터 품질** (`/quality-assessments`)
    같은 데이터에 대해 결측·이상치·라벨오류·불균형·잡음·중복을 재고,
    점수와 학습 영향으로 환산한 뒤,
    "이 데이터가 쓸 만한가"를 판정한다.

**모듈 3 — 모델** (`/training-runs`)
    두 게이트를 통과한 데이터로 모델을 학습시키고,
    곡선·혼동 행렬·지연시간을 근거로
    "이 모델이 현장에 나가도 되는가"를 판정한다.

**모듈 4 — 최적화** (`/optimization-runs`)
    승인받은 모델을 실행 경로(TorchScript / ONNX / TFLite)와
    정밀도(FP32 / FP16 / INT8)로 바꿔 보고,
    변환 전후가 같은 답을 내는지 대조하고, 속도·크기·정확도를 한 표에 놓은 뒤,
    "디바이스 예산 안에서 무엇을 쓸 것인가"를 판정한다.

**모듈 5 — 운영** (`/deployments`, `/health-watches`)
    승인·선택된 결과물을 현장에 올리고, 판단을 전부 로그로 남기고,
    창 단위로 지연시간·예측분포·입력분포를 지켜본 뒤,
    "이것이 아직 괜찮은가"를 계속 묻는다.
    이상하면 격리하고, 필요하면 롤백하고, 결국 재학습으로 되돌아간다.

**모듈 6 — AWS** (`/fleets`, `/rollouts`)
    수천 대의 데이터를 클라우드로 모으고, 그것으로 다시 학습하고,
    새 모델을 단계적으로 현장에 내보내고, 문제가 생기면 되돌린다.
    그리고 계보로 **Edge → Cloud → Edge 가 이어져 있음**을 증명한다.

앞의 네 게이트는 **내보내기 전에** 묻고, 모듈 5 는 **내보낸 뒤에 계속** 묻는다.
그리고 현장에는 정답이 없으므로, 정확도 대신 잴 수 있는 것으로 답한다.
모듈 6 은 그 순환을 수천 대 규모로 닫는다.

학습은 오래 걸리는 작업이므로 요청이 그것을 붙잡고 기다리지 않는다 (§11).
`POST /training-runs/{id}/start` 는 202 를 즉시 돌려주고,
진행 상황은 `GET /training-runs/{id}` 로 물어본다.

실습 문서: `docs/curriculum/`
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="On-Device AI Backend",
        version="0.1.0",
        description=DESCRIPTION,
    )
    register_exception_handlers(app)
    app.include_router(datasets.router)
    app.include_router(quality.router)
    app.include_router(models.router)
    app.include_router(optimization.router)
    app.include_router(operations.router)
    app.include_router(operations.watch_router)
    app.include_router(fleet.router)
    app.include_router(fleet.rollout_router)
    app.include_router(fleet.training_router)
    # 확장 실습 (1-11, 2-11, 3-11~3-15, 4-11~4-14, 6-12~6-14)
    app.include_router(extended.data_router)
    app.include_router(extended.model_router)
    app.include_router(extended.optimization_router)
    app.include_router(extended.experiment_router)
    app.include_router(extended.storage_router)
    app.include_router(extended.endpoint_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
