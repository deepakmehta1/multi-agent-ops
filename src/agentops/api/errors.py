"""Maps domain exceptions onto HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agentops.core.exceptions import AgentOpsError
from agentops.core.logging import get_logger

log = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AgentOpsError)
    async def _handle_known(request: Request, exc: AgentOpsError) -> JSONResponse:
        log.warning(
            "api.error",
            extra={"code": exc.code, "path": request.url.path, "detail": exc.message},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "code": exc.code, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def _handle_unknown(request: Request, exc: Exception) -> JSONResponse:
        log.exception("api.unhandled", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": "An unexpected error occurred.",
                "code": "internal_error",
                "details": {},
            },
        )
