"""The orchestrator: plan, delegate, review, answer.

One request becomes:

    plan -> execute the graph -> (optionally) revise and execute again -> synthesize

Everything is traced, bounded by a wall-clock budget, and degrades to `partial`
rather than failing outright when some specialists succeed and others do not.
"""

from __future__ import annotations

import asyncio
import uuid
from time import perf_counter

from agentops.agents.base import RunContext
from agentops.agents.registry import AgentRegistry
from agentops.core.config import Settings
from agentops.core.exceptions import AgentOpsError
from agentops.core.logging import get_logger
from agentops.llm.base import LLMClient, Usage
from agentops.orchestrator.executor import Executor
from agentops.orchestrator.models import (
    Plan,
    RunResult,
    RunStatus,
    StepResult,
    StepStatus,
    UsageView,
)
from agentops.orchestrator.planner import Planner
from agentops.orchestrator.synthesizer import Synthesizer
from agentops.runtime.events import EventBus, EventType
from agentops.tools.base import ToolRegistry

log = get_logger(__name__)


class Orchestrator:
    """The Delivery Lead: plans a request, delegates it, reviews the results and answers.

    Holds the long-lived collaborators (model client, roster, tools, settings). Every call to
    `run` builds fresh per-run state, so one instance safely serves concurrent requests.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        agents: AgentRegistry,
        tools: ToolRegistry,
        settings: Settings,
    ) -> None:
        """Create an orchestrator over a fixed set of collaborators.

        Args:
            llm: The model client used for planning, every agent turn and synthesis.
            agents: The roster of specialists.
            tools: The tool registry the specialists draw from.
            settings: Limits and model names.
        """
        self._llm = llm
        self._agents = agents
        self._tools = tools
        self._settings = settings
        self._executor = Executor(agents, settings)

    @staticmethod
    def new_run_id() -> str:
        """A new run id: `run_` followed by 12 random hex characters."""
        return f"run_{uuid.uuid4().hex[:12]}"

    async def run(
        self,
        request: str,
        *,
        run_id: str | None = None,
        bus: EventBus | None = None,
    ) -> RunResult:
        """Handle one request end to end and return the result. Never raises for ordinary failures.

        Wraps the whole run in `run_timeout_seconds`. A provider error, a plan the model could
        not get right, or a timeout all come back as a result with status `failed` and an
        explanatory answer, so callers always get a `RunResult` to store and show. Only
        cancellation propagates, which is what lets a client hang-up stop the work.

        Whatever the outcome, this method fills in the totals (duration, tokens, LLM calls) and
        the trace, then closes the event bus so streaming subscribers know it is over.

        Args:
            request: What the team asked, in plain language.
            run_id: Use this id instead of generating one (the streaming endpoint needs the id
                before the run finishes).
            bus: Use this trace bus instead of a fresh one, so a caller can subscribe first.
        """
        run_id = run_id or self.new_run_id()
        bus = bus or EventBus(run_id)
        started = perf_counter()
        planner = Planner(self._llm, self._agents, self._settings)
        synthesizer = Synthesizer(self._llm, self._settings)

        bus.emit(EventType.RUN_STARTED, agent="orchestrator", message=request)

        try:
            result = await asyncio.wait_for(
                self._run_inner(request, run_id, bus, planner, synthesizer),
                timeout=self._settings.run_timeout_seconds,
            )
        # Whole-run budget exceeded; wait_for has cancelled the work in flight.
        except TimeoutError:
            result = self._failure(
                run_id,
                request,
                f"The run exceeded its {self._settings.run_timeout_seconds:.0f}s budget.",
                bus,
            )
        # A deliberate, expected failure (provider down, plan unrecoverable).
        except AgentOpsError as exc:
            log.warning("run.failed", extra={"run_id": run_id, "error": exc.message})
            result = self._failure(run_id, request, exc.message, bus)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("run.crashed", extra={"run_id": run_id})
            result = self._failure(run_id, request, f"{type(exc).__name__}: {exc}", bus)

        result.duration_ms = int((perf_counter() - started) * 1000)
        # Usage is accounted for here and nowhere else, so that every exit path --
        # a delegated run, a direct answer, a partial result, a failure, a timeout --
        # reports the tokens that were really spent, planning included.
        result.usage = _usage_for(result.steps, planner.usage + synthesizer.usage)
        result.usage.llm_calls = (
            planner.llm_calls
            + synthesizer.llm_calls
            + sum(step.iterations for step in result.steps)
        )
        result.trace = list(bus.history)
        # Always last: tells streaming subscribers the trace is complete.
        bus.close()
        return result

    async def _run_inner(
        self,
        request: str,
        run_id: str,
        bus: EventBus,
        planner: Planner,
        synthesizer: Synthesizer,
    ) -> RunResult:
        """The run itself: plan, delegate, review, synthesize.

        1. Plan. If the plan is a direct answer, return it - no specialist, no synthesis.
        2. Execute the plan's graph.
        3. Review, up to `max_replan_rounds` times: ask the planner whether anything is missing
           and execute any follow-up steps (which may depend on earlier results).
        4. Synthesize one answer from everything that ran.

        Raises:
            AgentOpsError: Planning failed or the model provider failed; `run` converts this
                into a failed result.
        """
        ctx = RunContext(
            run_id=run_id,
            llm=self._llm,
            registry=self._tools,
            bus=bus,
            settings=self._settings,
        )

        plan = await planner.plan(request, bus)
        # The Delivery Lead decided no specialist is needed. Skip execution, review and synthesis.
        if plan.is_direct:
            answer = plan.answer_directly or ""
            bus.emit(
                EventType.RUN_COMPLETED,
                agent="orchestrator",
                message="Answered without delegating",
                delegated=False,
            )
            return RunResult(
                run_id=run_id,
                request=request,
                status=RunStatus.SUCCEEDED,
                answer=answer,
                plan=plan,
            )

        steps = await self._executor.execute(plan, request, ctx)
        revisions: list[Plan] = []

        # Review rounds. Bounded by config so a model that keeps finding "one more thing" cannot
        # loop forever; a round that adds nothing ends the loop early.
        for _ in range(self._settings.max_replan_rounds):
            revision = await planner.replan(request, plan, steps, bus)
            # Empty revision = the planner is satisfied; stop reviewing.
            if not revision.steps:
                break
            revisions.append(revision)
            # `prior=steps` lets the follow-up depend on earlier results.
            steps += await self._executor.execute(revision, request, ctx, prior=steps)

        bus.emit(
            EventType.SYNTHESIS_STARTED,
            agent="orchestrator",
            message=f"Combining {len(steps)} specialist report(s)",
        )
        answer, _ = await synthesizer.synthesize(request, steps)

        status = _status_for(steps)
        bus.emit(
            EventType.RUN_COMPLETED,
            agent="orchestrator",
            message=f"Run {status}",
            steps=len(steps),
            delegated=True,
        )
        return RunResult(
            run_id=run_id,
            request=request,
            status=status,
            answer=answer,
            plan=plan,
            revisions=revisions,
            steps=steps,
        )

    def _failure(self, run_id: str, request: str, message: str, bus: EventBus) -> RunResult:
        """Build the result for a run that could not complete, and emit `run.failed`.

        The answer text explains what happened so the person asking is never left with a blank.
        """
        bus.emit(EventType.RUN_FAILED, agent="orchestrator", message=message)
        return RunResult(
            run_id=run_id,
            request=request,
            status=RunStatus.FAILED,
            answer=f"This request could not be completed: {message}",
            plan=Plan(reasoning="Run failed before a plan could be executed."),
        )


def _status_for(steps: list[StepResult]) -> RunStatus:
    """Derive the run status from its steps.

    All succeeded: succeeded. Some did: partial. None did, or there were no steps: failed.
    """
    if not steps:
        return RunStatus.FAILED
    if all(step.status is StepStatus.SUCCEEDED for step in steps):
        return RunStatus.SUCCEEDED
    if any(step.status is StepStatus.SUCCEEDED for step in steps):
        return RunStatus.PARTIAL
    return RunStatus.FAILED


def _usage_for(steps: list[StepResult], extra: Usage) -> UsageView:
    """Total the tokens spent by every step plus the planner and synthesizer (`extra`).

    `llm_calls` is filled in separately by `Orchestrator.run`.
    """
    prompt = sum(step.prompt_tokens for step in steps) + extra.prompt_tokens
    completion = sum(step.completion_tokens for step in steps) + extra.completion_tokens
    return UsageView(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )
