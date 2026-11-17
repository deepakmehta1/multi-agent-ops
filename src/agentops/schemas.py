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
    name: str
    title: str
    mandate: str
    tools: list[str]


class AgentsResponse(BaseModel):
    orchestrator: dict[str, Any]
    agents: list[AgentView]


class ToolView(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    used_by: list[str]


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    llm_credentials: str
    agents: int
    tools: int


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: dict[str, Any] = Field(default_factory=dict)
