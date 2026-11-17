"""The data the orchestrator plans with and reports back."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from agentops.runtime.events import TraceEvent


class StepStatus(StrEnum):
    """Outcome of one delegation: it worked, it failed, or it never ran."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(StrEnum):
    """Outcome of a whole run.

    "partial" means at least one specialist succeeded and at least one did not - the answer
    is built from what worked and says what is unverified.
    """

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
    """A tool call as exposed in a run result (arguments and a truncated result)."""

    name: str
    arguments: dict = Field(default_factory=dict)
    ok: bool = True
    duration_ms: int = 0
    result_preview: str = ""


class StepResult(BaseModel):
    """The outcome of one plan step: its report or error, how it ran, and what it cost."""

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
        """Convenience for `status is SUCCEEDED`; used to decide dependants."""
        return self.status is StepStatus.SUCCEEDED


class UsageView(BaseModel):
    """Token and call totals for a whole run, covering planning, every agent turn and synthesis."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


class RunResult(BaseModel):
    """Everything one run produced: the answer, the plan(s), per-step results, trace and cost."""

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
        """Names of agents that ran, in first-use order, without repeats."""
        seen: list[str] = []
        for step in self.steps:
            if step.agent not in seen:
                seen.append(step.agent)
        return seen
