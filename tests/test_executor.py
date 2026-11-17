from __future__ import annotations

import asyncio

from agentops.agents.base import RunContext
from agentops.core.exceptions import LLMError
from agentops.llm.base import Completion, Message, Usage
from agentops.orchestrator.executor import Executor
from agentops.orchestrator.models import Plan, PlanStep, StepStatus
from agentops.runtime.events import EventType
from helpers import system_prompt


class RecordingLLM:
    """Returns a fixed answer, remembers the prompts, optionally stalls or fails."""

    def __init__(self, *, delay: float = 0.0, fail_for: str | None = None) -> None:
        self.delay = delay
        self.fail_for = fail_for
        self.prompts: list[str] = []

    async def complete(self, messages, *, model, tools=None, temperature=None, json_object=False):
        prompt = "\n".join(m.content or "" for m in messages)
        self.prompts.append(prompt)
        if self.fail_for and self.fail_for in system_prompt(messages):
            raise LLMError("provider exploded")
        if self.delay:
            await asyncio.sleep(self.delay)
        return Completion(
            message=Message(role="assistant", content="report body"),
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            model=model,
        )


def _ctx(llm, tools, bus, settings) -> RunContext:
    return RunContext(run_id="run_test", llm=llm, registry=tools, bus=bus, settings=settings)


def _plan(*steps: PlanStep) -> Plan:
    return Plan(reasoning="t", steps=list(steps))


async def test_independent_steps_run_concurrently(agents, tools, bus, settings):
    llm = RecordingLLM(delay=0.05)
    plan = _plan(
        PlanStep(id="s1", agent="code_reviewer", objective="Review PR 482."),
        PlanStep(id="s2", agent="test_engineer", objective="Assess coverage."),
        PlanStep(id="s3", agent="release_manager", objective="Check readiness."),
    )

    started = asyncio.get_running_loop().time()
    results = await Executor(agents, settings).execute(plan, "req", _ctx(llm, tools, bus, settings))
    elapsed = asyncio.get_running_loop().time() - started

    assert {r.status for r in results} == {StepStatus.SUCCEEDED}
    assert elapsed < 0.15, "three 50ms steps should overlap, not queue up"

    # Every step starts before any of them finishes.
    milestones = {EventType.STEP_STARTED, EventType.STEP_COMPLETED}
    kinds = [e.type for e in bus.history if e.type in milestones]
    assert kinds[:3] == [EventType.STEP_STARTED] * 3


async def test_parallelism_is_capped(agents, tools, bus, settings):
    llm = RecordingLLM(delay=0.05)
    capped = settings.model_copy(update={"max_parallel_steps": 1})
    plan = _plan(
        PlanStep(id="s1", agent="code_reviewer", objective="Review."),
        PlanStep(id="s2", agent="test_engineer", objective="Test."),
    )

    started = asyncio.get_running_loop().time()
    await Executor(agents, capped).execute(plan, "req", _ctx(llm, tools, bus, capped))
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed >= 0.1, "with a cap of one, the steps must serialise"


async def test_a_dependent_step_receives_the_upstream_output(agents, tools, bus, settings):
    llm = RecordingLLM()
    plan = _plan(
        PlanStep(id="s1", agent="incident_responder", objective="Investigate INC-204."),
        PlanStep(id="s2", agent="release_manager", objective="Decide go/no-go.", depends_on=["s1"]),
    )

    results = await Executor(agents, settings).execute(plan, "req", _ctx(llm, tools, bus, settings))

    assert [r.step_id for r in results] == ["s1", "s2"]
    downstream = next(p for p in llm.prompts if "Release Manager" in p)
    assert "From incident_responder" in downstream
    assert "report body" in downstream


async def test_a_failed_step_skips_everything_downstream(agents, tools, bus, settings):
    llm = RecordingLLM(fail_for="Site Reliability Engineer")
    plan = _plan(
        PlanStep(id="s1", agent="incident_responder", objective="Investigate."),
        PlanStep(id="s2", agent="release_manager", objective="Decide.", depends_on=["s1"]),
        PlanStep(id="s3", agent="code_reviewer", objective="Review PR 482."),
    )

    results = {
        r.step_id: r
        for r in await Executor(agents, settings).execute(
            plan, "req", _ctx(llm, tools, bus, settings)
        )
    }

    assert results["s1"].status is StepStatus.FAILED
    assert "provider exploded" in (results["s1"].error or "")
    assert results["s2"].status is StepStatus.SKIPPED
    assert "did not succeed" in (results["s2"].error or "")
    # An independent step is unaffected by an unrelated failure.
    assert results["s3"].status is StepStatus.SUCCEEDED


async def test_a_slow_step_hits_the_timeout_without_killing_the_run(agents, tools, bus, settings):
    impatient = settings.model_copy(update={"step_timeout_seconds": 0.05})
    llm = RecordingLLM(delay=0.5)
    plan = _plan(PlanStep(id="s1", agent="code_reviewer", objective="Review PR 482."))

    results = await Executor(agents, impatient).execute(
        plan, "req", _ctx(llm, tools, bus, impatient)
    )

    assert results[0].status is StepStatus.FAILED
    assert "step timeout" in (results[0].error or "")


async def test_a_follow_up_plan_can_depend_on_an_earlier_round(agents, tools, bus, settings):
    llm = RecordingLLM()
    executor = Executor(agents, settings)
    first = await executor.execute(
        _plan(PlanStep(id="s1", agent="code_reviewer", objective="Review PR 482.")),
        "req",
        _ctx(llm, tools, bus, settings),
    )
    second = await executor.execute(
        _plan(
            PlanStep(
                id="s2",
                agent="test_engineer",
                objective="Write the missing tests.",
                depends_on=["s1"],
            )
        ),
        "req",
        _ctx(llm, tools, bus, settings),
        prior=first,
    )

    assert second[0].status is StepStatus.SUCCEEDED
    downstream = next(p for p in llm.prompts if "Engineer in Test" in p)
    assert "From code_reviewer" in downstream
