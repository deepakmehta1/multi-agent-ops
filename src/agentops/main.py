"""FastAPI application factory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from agentops import __version__
from agentops.agents.registry import AgentRegistry
from agentops.api.errors import register_error_handlers
from agentops.api.routes_agents import router as agents_router
from agentops.api.routes_health import router as health_router
from agentops.api.routes_runs import router as runs_router
from agentops.core.config import Settings, get_settings
from agentops.core.logging import configure_logging, get_logger
from agentops.runtime.store import InMemoryRunStore
from agentops.tools.catalog import build_registry

log = get_logger(__name__)

DESCRIPTION = """
A multi-agent system for a software delivery team.

One **orchestrator** (the Delivery Lead) reads a request, breaks it into a dependency
graph of delegations, runs the independent ones in parallel, reviews what comes back,
and writes a single answer. The **specialists** -- code reviewer, test engineer,
incident responder, release manager -- each have their own prompt, their own tools and
their own bounded tool-calling loop.

Every run is traced: `POST /v1/requests/stream` shows the plan, each delegation and
each tool call as it happens.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.agents = AgentRegistry()
        app.state.tools = build_registry()
        app.state.runs = InMemoryRunStore(max_size=settings.run_history_size)
        app.state.llm_cache = {}
        log.info(
            "service.started",
            extra={
                "environment": settings.environment,
                "agents": len(app.state.agents),
                "tools": len(app.state.tools),
                "orchestrator_model": settings.orchestrator_model,
                "agent_model": settings.agent_model,
                "server_key": settings.has_server_key,
            },
        )
        yield
        for client in app.state.llm_cache.values():
            await client.aclose()
        log.info("service.stopped")

    app = FastAPI(
        title="multi-agent-ops",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(agents_router)
    app.include_router(runs_router)
    return app


app = create_app()
