"""The planner: plan shape, validation and the single repair attempt."""

from __future__ import annotations

import json

import pytest

from agentops.core.exceptions import PlanValidationError
from agentops.llm.fake import Reply, ScriptedLLM
from agentops.orchestrator.planner import Planner
from agentops.runtime.events import EventType
from helpers import direct_reply, is_first_plan, plan_reply, step


async def test_independent_work_is_planned_without_dependencies(agents, settings, bus):
    """Independent steps come back with no dependencies, and plan.created is emitted."""
    llm = ScriptedLLM()
    llm.when(
        is_first_plan,
        plan_reply(
            step("s1", "code_reviewer", "Review PR 482."),
            step("s2", "test_engineer", "Assess coverage for PR 482."),
        ),
    )
    plan = await Planner(llm, agents, settings).plan("Review PR 482 and its tests", bus)

    assert [s.agent for s in plan.steps] == ["code_reviewer", "test_engineer"]
    assert all(not s.depends_on for s in plan.steps)
    assert bus.history[0].type is EventType.PLAN_CREATED


async def test_a_dependency_is_preserved(agents, settings, bus):
    """A dependency the model declares survives into the plan."""
    llm = ScriptedLLM()
    llm.when(
        is_first_plan,
        plan_reply(
            step("s1", "incident_responder", "Investigate INC-204."),
            step("s2", "release_manager", "Decide go/no-go.", ["s1"]),
        ),
    )
    plan = await Planner(llm, agents, settings).plan("Can we ship with INC-204 open?", bus)
    assert plan.steps[1].depends_on == ["s1"]


async def test_a_simple_question_is_answered_without_delegating(agents, settings, bus):
    """A direct answer yields a plan with no steps that reports `is_direct`."""
    llm = ScriptedLLM()
    llm.when(is_first_plan, direct_reply("The release train runs Thursdays at 14:00 UTC."))
    plan = await Planner(llm, agents, settings).plan("When does the release train run?", bus)

    assert plan.is_direct
    assert not plan.steps
    assert "Thursdays" in (plan.answer_directly or "")


async def test_an_unknown_agent_is_rejected_and_repaired(agents, settings, bus):
    """A plan naming a non-existent agent is rejected, then fixed on the one repair attempt."""
    llm = ScriptedLLM()
    llm.when(is_first_plan, plan_reply(step("s1", "security_auditor", "Audit the code.")))
    llm.when(
        lambda messages, _tools: any("was rejected" in (m.content or "") for m in messages),
        plan_reply(step("s1", "code_reviewer", "Review PR 482.")),
    )

    plan = await Planner(llm, agents, settings).plan("Audit PR 482", bus)

    assert plan.steps[0].agent == "code_reviewer"
    assert len(llm.calls) == 2, "the planner should get exactly one repair attempt"


async def test_a_plan_that_stays_invalid_fails_the_run(agents, settings, bus):
    """If the repair is also invalid, planning raises rather than looping."""
    llm = ScriptedLLM()
    llm.when(
        lambda _m, tool_specs: not tool_specs,
        plan_reply(step("s1", "nobody", "Do something.")),
        sticky=True,
    )
    with pytest.raises(PlanValidationError, match="unknown agent"):
        await Planner(llm, agents, settings).plan("Do something", bus)


async def test_a_dependency_cycle_is_rejected(agents, settings, bus):
    """Two steps that depend on each other are rejected."""
    llm = ScriptedLLM()
    llm.when(
        lambda _m, tool_specs: not tool_specs,
        plan_reply(
            step("s1", "code_reviewer", "Review.", ["s2"]),
            step("s2", "test_engineer", "Test.", ["s1"]),
        ),
        sticky=True,
    )
    with pytest.raises(PlanValidationError, match="cycle"):
        await Planner(llm, agents, settings).plan("Review and test", bus)


async def test_a_dangling_dependency_is_rejected(agents, settings, bus):
    """A dependency on a step that does not exist is rejected."""
    llm = ScriptedLLM()
    llm.when(
        lambda _m, tool_specs: not tool_specs,
        plan_reply(step("s1", "code_reviewer", "Review.", ["s9"])),
        sticky=True,
    )
    with pytest.raises(PlanValidationError, match="unknown step"):
        await Planner(llm, agents, settings).plan("Review", bus)


async def test_the_step_budget_is_enforced(agents, settings, bus):
    """A plan over `max_plan_steps` is rejected, whatever the model intended."""
    llm = ScriptedLLM()
    too_many = [
        step(f"s{i}", "code_reviewer", f"Review {i}.") for i in range(settings.max_plan_steps + 1)
    ]
    llm.when(lambda _m, tool_specs: not tool_specs, plan_reply(*too_many), sticky=True)
    with pytest.raises(PlanValidationError, match="the limit is"):
        await Planner(llm, agents, settings).plan("Review everything", bus)


async def test_malformed_json_is_rejected(agents, settings, bus):
    """A planner reply that is not JSON never reaches the executor.

    Note what this really proves: with `json_object=True` the scripted fake parses its own
    fixture and raises before the planner sees anything, so this tests the fake's guard, not
    the planner's `_parse`. The planner's own invalid-JSON, non-object and schema-mismatch
    branches, plus its empty-plan, duplicate-id, empty-objective and self-dependency checks,
    have no test yet.
    """
    llm = ScriptedLLM()
    llm.when(lambda _m, tool_specs: not tool_specs, Reply(content="not json at all"), sticky=True)
    # json_object=True makes the fake assert on its own fixture, which is the point:
    # a malformed plan can never silently reach the executor.
    with pytest.raises(json.JSONDecodeError):
        await Planner(llm, agents, settings).plan("Review", bus)


async def test_replanning_may_return_no_further_work(agents, settings, bus):
    """A review that finds nothing missing returns no steps and emits no plan.revised."""
    llm = ScriptedLLM()
    llm.when(is_first_plan, plan_reply(step("s1", "code_reviewer", "Review PR 482.")))
    llm.when(
        lambda messages, _tools: any(
            "Is anything still missing" in (m.content or "") for m in messages
        ),
        Reply(content=json.dumps({"reasoning": "done", "steps": []})),
    )
    planner = Planner(llm, agents, settings)
    plan = await planner.plan("Review PR 482", bus)
    revision = await planner.replan("Review PR 482", plan, [], bus)

    assert not revision.steps
    assert not any(event.type is EventType.PLAN_REVISED for event in bus.history)
