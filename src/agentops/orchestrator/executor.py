"""Runs a plan as a dependency graph.

Steps with no outstanding dependencies run concurrently, bounded by
`max_parallel_steps`. A step's output is handed to whatever depends on it. If a
step fails, everything downstream of it is skipped rather than run on missing
input -- the run then finishes as `partial` instead of collapsing.
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from agentops.agents.base import AgentTask, RunContext
from agentops.agents.registry import AgentRegistry
from agentops.core.config import Settings
from agentops.core.exceptions import AgentOpsError
from agentops.core.logging import get_logger
from agentops.orchestrator.models import (
    Plan,
    PlanStep,
    StepResult,
    StepStatus,
    ToolCallView,
)
from agentops.runtime.events import EventType

log = get_logger(__name__)


class Executor:
    """Runs a plan's steps as a dependency graph, in parallel where it can.

    Scheduling is by "waves": each pass takes every step whose dependencies are all finished
    and runs that set concurrently. A step in a later wave therefore waits for the WHOLE
    previous wave, even if its own dependencies finished sooner - a deliberate trade of a
    little latency for a much simpler scheduler, fine for plans of at most a handful of steps.
    """

    def __init__(self, agents: AgentRegistry, settings: Settings) -> None:
        """Bind the roster (to find each step's agent) and the settings (limits)."""
        self._agents = agents
        self._settings = settings

    async def execute(
        self,
        plan: Plan,
        request: str,
        ctx: RunContext,
        *,
        prior: list[StepResult] | None = None,
    ) -> list[StepResult]:
        """Execute every step of `plan`, honouring dependencies, and return the new results.

        Failure handling is the point of this method. A step that fails does not stop the run:
        it is recorded as failed, and every step that depends on it, directly or transitively,
        is recorded as skipped without being run - running a step on missing input would waste
        tokens and could produce a confident wrong answer. Independent branches carry on.

        Args:
            plan: The steps to run.
            request: The original request, passed to every agent for context.
            ctx: Per-run services.
            prior: Results from earlier rounds. A follow-up plan may depend on these; they are
                consulted but not returned again.

        Returns:
            One `StepResult` per step of `plan`, in the order they finished (never raises for
            a step failing; see `StepResult.status`).
        """
        results: dict[str, StepResult] = {r.step_id: r for r in (prior or [])}
        produced: list[StepResult] = []
        remaining = {step.id: step for step in plan.steps}
        # Caps how many agents run at once across the whole call. `max(1, ...)` stops a zero or
        # negative setting from deadlocking every step.
        semaphore = asyncio.Semaphore(max(1, self._settings.max_parallel_steps))

        # Each pass is one "wave". Every pass either runs or skips at least one step, so this loop
        # always terminates.
        while remaining:
            wave = [
                step
                for step in remaining.values()
                if all(dep in results for dep in step.depends_on)
            ]
            # Defensive: steps remain but none can ever become ready (a dependency that is neither
            # finished nor in this plan). Plan validation should make this impossible; if it ever
            # happens we skip them rather than spin forever.
            if not wave:
                # Dependencies that will never arrive (an upstream step was skipped).
                for step in remaining.values():
                    unmet = [dep for dep in step.depends_on if dep not in results]
                    skipped = _skipped(step, f"Upstream step(s) never completed: {unmet}")
                    results[step.id] = skipped
                    produced.append(skipped)
                    ctx.bus.emit(
                        EventType.STEP_SKIPPED,
                        agent=step.agent,
                        step_id=step.id,
                        message=skipped.error,
                    )
                break

            # Of the steps whose dependencies have finished, only those whose dependencies all
            # SUCCEEDED may run. The rest are skipped just below, and since a skipped step counts as
            # finished-but-not-succeeded, its own dependants are skipped on the next pass: failure
            # cascades down the graph and stops there.
            runnable = [
                step for step in wave if all(results[dep].succeeded for dep in step.depends_on)
            ]
            for step in wave:
                if step in runnable:
                    continue
                failed = [dep for dep in step.depends_on if not results[dep].succeeded]
                skipped = _skipped(step, f"Skipped because step(s) {failed} did not succeed.")
                results[step.id] = skipped
                produced.append(skipped)
                remaining.pop(step.id)
                ctx.bus.emit(
                    EventType.STEP_SKIPPED,
                    agent=step.agent,
                    step_id=step.id,
                    message=skipped.error,
                )

            # Everything in this wave was skipped; go round again, since that may unblock a skip.
            if not runnable:
                continue

            # Run the whole wave concurrently. `_run_step` never raises for step-level failures, so
            # one failing step cannot cancel its siblings.
            wave_results = await asyncio.gather(
                *(self._run_step(step, request, ctx, results, semaphore) for step in runnable)
            )
            for step, result in zip(runnable, wave_results, strict=True):
                results[step.id] = result
                produced.append(result)
                remaining.pop(step.id)

        return produced

    async def _run_step(
        self,
        step: PlanStep,
        request: str,
        ctx: RunContext,
        results: dict[str, StepResult],
        semaphore: asyncio.Semaphore,
    ) -> StepResult:
        """Run one step: build its task, run the agent under a timeout, and convert the outcome.

        Never raises for a step-level problem. A timeout, a domain error (such as a model
        provider failure) or an unexpected exception each become a `FAILED` result with a
        message, so one bad step cannot take its siblings down through `asyncio.gather`.
        Cancellation is the exception and propagates, so a stopped run really stops.

        Args:
            step: The plan step.
            request: The original request.
            ctx: Per-run services.
            results: Results so far; the outputs of `step.depends_on` are read from here.
            semaphore: Shared limiter for `max_parallel_steps`.
        """
        agent = self._agents.get(step.agent)
        # Hand each dependency's report to this step, labelled by the agent that wrote it.
        # LIMITATION: keyed by agent name, so two upstream steps run by the SAME agent would
        # overwrite each other and only the later report would be passed on. The planner prompt
        # discourages giving two steps to one agent for this reason; it is not enforced.
        context = {
            results[dep].agent: results[dep].output
            for dep in step.depends_on
            if dep in results and results[dep].output
        }
        task = AgentTask(objective=step.objective, request=request, context=context)

        ctx.bus.emit(
            EventType.STEP_STARTED,
            agent=agent.name,
            step_id=step.id,
            message=step.objective,
            depends_on=step.depends_on,
        )
        started = perf_counter()

        # Hold a slot only while the agent is actually working, not while building the task.
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    agent.run(task, ctx, step_id=step.id),
                    timeout=self._settings.step_timeout_seconds,
                )
            # wait_for has already cancelled the agent's work by the time this is raised.
            except TimeoutError:
                return self._failed(
                    step,
                    f"{agent.title} exceeded the {self._settings.step_timeout_seconds:.0f}s "
                    "step timeout.",
                    started,
                    ctx,
                )
            except AgentOpsError as exc:
                return self._failed(step, exc.message, started, ctx)
            # Catch-all so a bug in one agent fails that step, not the run. CancelledError is a
            # BaseException and deliberately NOT caught here.
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("step.crashed", extra={"run_id": ctx.run_id, "step_id": step.id})
                return self._failed(step, f"{type(exc).__name__}: {exc}", started, ctx)

        return StepResult(
            step_id=step.id,
            agent=agent.name,
            objective=step.objective,
            status=StepStatus.SUCCEEDED,
            output=result.output,
            tool_calls=[
                ToolCallView(
                    name=record.name,
                    arguments=record.arguments,
                    ok=record.ok,
                    duration_ms=record.duration_ms,
                    result_preview=record.result_preview,
                )
                for record in result.tool_calls
            ],
            iterations=result.iterations,
            truncated=result.truncated,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            duration_ms=int((perf_counter() - started) * 1000),
        )

    def _failed(self, step: PlanStep, error: str, started: float, ctx: RunContext) -> StepResult:
        """Emit `step.failed` and build the matching `FAILED` result."""
        ctx.bus.emit(EventType.STEP_FAILED, agent=step.agent, step_id=step.id, message=error)
        return StepResult(
            step_id=step.id,
            agent=step.agent,
            objective=step.objective,
            status=StepStatus.FAILED,
            error=error,
            duration_ms=int((perf_counter() - started) * 1000),
        )


def _skipped(step: PlanStep, reason: str) -> StepResult:
    """Build a `SKIPPED` result carrying the reason; the caller emits the event."""
    return StepResult(
        step_id=step.id,
        agent=step.agent,
        objective=step.objective,
        status=StepStatus.SKIPPED,
        error=reason,
    )
