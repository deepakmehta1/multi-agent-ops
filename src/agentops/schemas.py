"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentops.orchestrator.models import (
    Plan,
    RunResult,
    RunStatus,
    StepResult,
    UsageView,
)
from agentops.runtime.events import TraceEvent


class RunRequest(BaseModel):
    """Body of `POST /v1/requests` and `/v1/requests/stream`."""

    request: str = Field(
        min_length=3,
        max_length=8000,
        description="What the team needs, in plain language.",
        examples=["Review PR 482 on checkout-api and tell me if it is safe to merge."],
    )
    include_trace: bool = Field(
        default=False,
        description="Include the full event trace in the response.",
    )


class StepView(BaseModel):
    """One specialist's work in a response: what it was asked, how it went and what it said.

    A flattened, client-friendly form of the internal `StepResult`.
    """

    step_id: str
    agent: str
    objective: str
    status: str
    output: str
    error: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    iterations: int = 0
    truncated: bool = False
    duration_ms: int = 0

    @classmethod
    def from_result(cls, result: StepResult) -> StepView:
        """Convert an internal `StepResult`, keeping only the tool names called."""
        return cls(
            step_id=result.step_id,
            agent=result.agent,
            objective=result.objective,
            status=str(result.status),
            output=result.output,
            error=result.error,
            tools_called=[call.name for call in result.tool_calls],
            iterations=result.iterations,
            truncated=result.truncated,
            duration_ms=result.duration_ms,
        )


class RunResponse(BaseModel):
    """What a run returns: the answer, how it was produced, and what it cost.

    `trace` is only populated when asked for (`include_trace`), since it can be long.
    """

    run_id: str
    request: str
    status: RunStatus
    answer: str
    agents_used: list[str]
    plan: Plan
    revisions: list[Plan] = Field(default_factory=list)
    steps: list[StepView] = Field(default_factory=list)
    usage: UsageView
    duration_ms: int
    trace: list[TraceEvent] | None = None

    @classmethod
    def from_run(cls, run: RunResult, *, include_trace: bool = False) -> RunResponse:
        """Convert an internal `RunResult` into the API shape.

        Args:
            run: The finished run.
            include_trace: Attach the full event trace.
        """
        return cls(
            run_id=run.run_id,
            request=run.request,
            status=run.status,
            answer=run.answer,
            agents_used=run.agents_used,
            plan=run.plan,
            revisions=run.revisions,
            steps=[StepView.from_result(step) for step in run.steps],
            usage=run.usage,
            duration_ms=run.duration_ms,
            trace=run.trace if include_trace else None,
        )


class AgentView(BaseModel):
    """One specialist on the roster, as served by `GET /v1/agents`."""

    name: str
    title: str
    mandate: str
    tools: list[str]


class AgentsResponse(BaseModel):
    """The orchestrator's description plus every specialist."""

    orchestrator: dict[str, Any]
    agents: list[AgentView]


class ToolView(BaseModel):
    """One tool: its schema and the agents permitted to call it."""

    name: str
    description: str
    parameters: dict[str, Any]
    used_by: list[str]


class HealthResponse(BaseModel):
    """Liveness probe body."""

    status: str
    environment: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness probe body. `llm_credentials` is "server-key" or "caller-supplied"."""

    status: str
    llm_credentials: str
    agents: int
    tools: int


class ErrorResponse(BaseModel):
    """The uniform error body documented in OpenAPI for non-2xx responses."""

    error: str
    code: str
    details: dict[str, Any] = Field(default_factory=dict)
