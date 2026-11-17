"""Maps domain exceptions onto HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agentops.core.exceptions import AgentOpsError
from agentops.core.logging import get_logger

log = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Install the handlers that turn exceptions into the API's uniform error body.

    Every error response has the shape `{"error": str, "code": str, "details": object}`,
    so clients handle one format. Expected failures (`AgentOpsError`) carry their own status
    code and message; anything else is a bug and is reported as a generic 500.
    """

    @app.exception_handler(AgentOpsError)
    async def _handle_known(request: Request, exc: AgentOpsError) -> JSONResponse:
        """Render a deliberate `AgentOpsError` with its own status code, code and details."""
        log.warning(
            "api.error",
            # NOT `message`: that name is reserved by `LogRecord`, and passing it in `extra` raises
            # KeyError, which once turned every error response into a 500. See core/logging.py.
            extra={"code": exc.code, "path": request.url.path, "detail": exc.message},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "code": exc.code, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def _handle_unknown(request: Request, exc: Exception) -> JSONResponse:
        """Last resort for anything unexpected: log it with a traceback, tell the caller nothing.

        The response deliberately omits the exception text, which could leak internals.
        """
        log.exception("api.unhandled", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": "An unexpected error occurred.",
                "code": "internal_error",
                "details": {},
            },
        )
