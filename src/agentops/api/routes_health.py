"""Liveness and readiness."""

from __future__ import annotations

from fastapi import APIRouter

from agentops import __version__
from agentops.api.deps import AgentsDep, SettingsDep, ToolsDep
from agentops.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz(settings: SettingsDep) -> HealthResponse:
    """Liveness: the process is up and serving. Does no work, so it is safe to poll often."""
    return HealthResponse(status="ok", environment=settings.environment, version=__version__)


@router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
async def readyz(settings: SettingsDep, agents: AgentsDep, tools: ToolsDep) -> ReadinessResponse:
    """Ready as long as the roster is wired up.

    A missing server-side key is reported but not fatal: callers may bring their own.
    """
    return ReadinessResponse(
        status="ready",
        llm_credentials="server-key" if settings.has_server_key else "caller-supplied",
        agents=len(agents),
        tools=len(tools),
    )
