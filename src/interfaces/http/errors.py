"""예외 → HTTP 변환. (CLAUDE.md §12)

Domain 은 자신이 몇 번 상태 코드가 될지 모른다. 그 지식은 여기에만 있다.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from application.shared.errors import (
    ApplicationException,
    ConflictingRequest,
    ResourceNotFound,
    UnsupportedOperation,
)
from domain.shared.errors import (
    DomainException,
    EntityNotFound,
    IllegalStateTransition,
    InvariantViolation,
)
from infrastructure.errors import InfrastructureException, SourceUnreadable


def _body(code: str, message: str, subject: str | None) -> dict[str, object]:
    return {"error": {"code": code, "message": message, "subject": subject}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(EntityNotFound)
    async def _entity_not_found(_: Request, exc: EntityNotFound) -> JSONResponse:
        # Dataset 이든 QualityAssessment 든, 없는 것은 없는 것이다.
        return JSONResponse(
            status_code=404, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(IllegalStateTransition)
    async def _illegal_state(_: Request, exc: IllegalStateTransition) -> JSONResponse:
        # 요청 자체는 형식이 맞지만 지금 순서에서는 할 수 없는 일이다 → 409
        return JSONResponse(
            status_code=409, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(InvariantViolation)
    async def _invariant(_: Request, exc: InvariantViolation) -> JSONResponse:
        return JSONResponse(
            status_code=422, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(DomainException)
    async def _domain(_: Request, exc: DomainException) -> JSONResponse:
        return JSONResponse(
            status_code=422, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(ResourceNotFound)
    async def _resource_not_found(_: Request, exc: ResourceNotFound) -> JSONResponse:
        return JSONResponse(
            status_code=404, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(ConflictingRequest)
    async def _conflict(_: Request, exc: ConflictingRequest) -> JSONResponse:
        return JSONResponse(
            status_code=409, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(UnsupportedOperation)
    async def _unsupported(_: Request, exc: UnsupportedOperation) -> JSONResponse:
        return JSONResponse(
            status_code=409, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(ApplicationException)
    async def _application(_: Request, exc: ApplicationException) -> JSONResponse:
        return JSONResponse(
            status_code=400, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(SourceUnreadable)
    async def _source_unreadable(_: Request, exc: SourceUnreadable) -> JSONResponse:
        # 데이터 원본을 읽지 못한 것은 클라이언트가 가리킨 위치의 문제다.
        return JSONResponse(
            status_code=422, content=_body(exc.code, exc.message, exc.subject)
        )

    @app.exception_handler(InfrastructureException)
    async def _infrastructure(_: Request, exc: InfrastructureException) -> JSONResponse:
        return JSONResponse(
            status_code=502, content=_body(exc.code, exc.message, exc.subject)
        )
