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
    def __init__(self, agents: AgentRegistry, settings: Settings) -> None:
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
        """Execute every step of `plan`, honouring dependencies.

        `prior` carries results from an earlier round so a follow-up plan can
        depend on steps that have already run.
        """
        results: dict[str, StepResult] = {r.step_id: r for r in (prior or [])}
        produced: list[StepResult] = []
        remaining = {step.id: step for step in plan.steps}
        semaphore = asyncio.Semaphore(max(1, self._settings.max_parallel_steps))

        while remaining:
            wave = [
                step
                for step in remaining.values()
                if all(dep in results for dep in step.depends_on)
            ]
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

            if not runnable:
                continue

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
        agent = self._agents.get(step.agent)
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

        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    agent.run(task, ctx, step_id=step.id),
                    timeout=self._settings.step_timeout_seconds,
                )
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
    return StepResult(
        step_id=step.id,
        agent=step.agent,
        objective=step.objective,
        status=StepStatus.SKIPPED,
        error=reason,
    )
