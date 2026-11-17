"""The data the orchestrator plans with and reports back."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from agentops.runtime.events import TraceEvent


class StepStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class PlanStep(BaseModel):
    """One delegation: give this agent this objective, after these other steps."""

    id: str = Field(description="Short identifier, e.g. 's1'.")
    agent: str = Field(description="Name of the specialist agent to delegate to.")
    objective: str = Field(description="What that agent must produce, in one or two sentences.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of steps whose output this step needs. Empty means it can start now.",
    )


class Plan(BaseModel):
    """The orchestrator's decision about how to handle one request."""

    reasoning: str = Field(default="", description="Why this breakdown.")
    answer_directly: str | None = Field(
        default=None,
        description="Set when no specialist is needed and the orchestrator can answer itself.",
    )
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def is_direct(self) -> bool:
        """True when the orchestrator answered without delegating to anyone."""
        return not self.steps and bool((self.answer_directly or "").strip())


class ToolCallView(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    ok: bool = True
    duration_ms: int = 0
    result_preview: str = ""


class StepResult(BaseModel):
    step_id: str
    agent: str
    objective: str
    status: StepStatus
    output: str = ""
    error: str | None = None
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    iterations: int = 0
    truncated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status is StepStatus.SUCCEEDED


class UsageView(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


class RunResult(BaseModel):
    run_id: str
    request: str
    status: RunStatus
    answer: str
    plan: Plan
    revisions: list[Plan] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    usage: UsageView = Field(default_factory=UsageView)
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def agents_used(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            if step.agent not in seen:
                seen.append(step.agent)
        return seen
