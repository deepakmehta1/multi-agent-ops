from __future__ import annotations

from agentops.llm.fake import Reply, ScriptedLLM
from agentops.orchestrator.models import RunStatus, StepStatus
from agentops.orchestrator.orchestrator import Orchestrator
from agentops.runtime.events import EventType
from helpers import (
    after_tool_result,
    agent_turn,
    direct_reply,
    is_first_plan,
    is_replan,
    is_synthesis,
    no_followup,
    plan_reply,
    step,
)


def _orchestrator(llm, agents, tools, settings) -> Orchestrator:
    return Orchestrator(llm=llm, agents=agents, tools=tools, settings=settings)


async def test_a_request_fans_out_and_comes_back_as_one_answer(agents, tools, settings):
    llm = ScriptedLLM()
    llm.when(
        is_first_plan,
        plan_reply(
            step("s1", "code_reviewer", "Review PR 482 on checkout-api."),
            step("s2", "test_engineer", "Assess test coverage for PR 482."),
        ),
    )
    llm.when(agent_turn("code_reviewer"), Reply(tool_calls=[("get_pull_request", {"number": 482})]))
    llm.when(
        agent_turn("test_engineer"),
        Reply(tool_calls=[("get_coverage_report", {"repo": "checkout-api"})]),
    )
    llm.when(after_tool_result, Reply(content="specialist report"), sticky=True)
    llm.when(is_replan, no_followup(), sticky=True)
    llm.when(is_synthesis, Reply(content="**Verdict:** do not merge yet."), sticky=True)

    result = await _orchestrator(llm, agents, tools, settings).run("Review PR 482 and its tests")

    assert result.status is RunStatus.SUCCEEDED
    assert result.answer == "**Verdict:** do not merge yet."
    assert result.agents_used == ["code_reviewer", "test_engineer"]
    assert all(s.status is StepStatus.SUCCEEDED for s in result.steps)
    assert result.usage.total_tokens > 0
    assert result.duration_ms >= 0

    kinds = [event.type for event in result.trace]
    assert kinds[0] is EventType.RUN_STARTED
    assert kinds[-1] is EventType.RUN_COMPLETED
    assert EventType.PLAN_CREATED in kinds
    assert EventType.SYNTHESIS_STARTED in kinds


async def test_a_single_specialist_answer_is_passed_through_verbatim(agents, tools, settings):
    """One report needs no synthesis -- re-writing it would only lose detail."""
    llm = ScriptedLLM()
    llm.when(is_first_plan, plan_reply(step("s1", "code_reviewer", "Review PR 482.")))
    llm.when(agent_turn("code_reviewer"), Reply(content="**Verdict:** request-changes\n\nDetail."))
    llm.when(is_replan, no_followup(), sticky=True)
    llm.when(is_synthesis, Reply(content="SHOULD NOT BE USED"), sticky=True)

    result = await _orchestrator(llm, agents, tools, settings).run("Review PR 482")

    assert result.answer.startswith("**Verdict:** request-changes")


async def test_a_trivial_question_skips_the_specialists_entirely(agents, tools, settings):
    llm = ScriptedLLM()
    llm.when(is_first_plan, direct_reply("The release train runs Thursdays at 14:00 UTC."))

    result = await _orchestrator(llm, agents, tools, settings).run("When is the release train?")

    assert result.status is RunStatus.SUCCEEDED
    assert result.steps == []
    assert "Thursdays" in result.answer
    assert len(llm.calls) == 1, "no specialist or synthesis call should happen"


async def test_the_orchestrator_can_schedule_follow_up_work(agents, tools, settings):
    llm = ScriptedLLM()
    llm.when(is_first_plan, plan_reply(step("s1", "code_reviewer", "Review PR 482.")))
    llm.when(
        is_replan,
        plan_reply(
            step("s2", "test_engineer", "Write the tests the review asked for.", ["s1"]),
            reasoning="the review exposed a coverage gap",
        ),
    )
    llm.when(agent_turn("code_reviewer"), Reply(content="Missing a concurrency test."), sticky=True)
    llm.when(agent_turn("test_engineer"), Reply(content="Added the test."), sticky=True)
    llm.when(is_synthesis, Reply(content="Reviewed, and the test is written."), sticky=True)

    result = await _orchestrator(llm, agents, tools, settings).run("Review PR 482")

    assert len(result.revisions) == 1
    assert [s.step_id for s in result.steps] == ["s1", "s2"]
    assert any(event.type is EventType.PLAN_REVISED for event in result.trace)


async def test_a_partly_failed_run_still_returns_what_worked(agents, tools, settings):
    class HalfBroken(ScriptedLLM):
        async def complete(self, messages, **kwargs):
            # Only the SRE's own turns fail; the planner prompt merely names the role.
            if "You are the Site Reliability Engineer" in (messages[0].content or ""):
                raise RuntimeError("provider exploded")
            return await super().complete(messages, **kwargs)

    llm = HalfBroken()
    llm.when(
        is_first_plan,
        plan_reply(
            step("s1", "incident_responder", "Investigate INC-204."),
            step("s2", "release_manager", "Check the release checklist."),
        ),
    )
    llm.when(
        agent_turn("release_manager"),
        Reply(content="Checklist has two blockers."),
        sticky=True,
    )
    llm.when(is_replan, no_followup(), sticky=True)
    llm.when(is_synthesis, Reply(content="Partial: the incident review did not run."), sticky=True)

    result = await _orchestrator(llm, agents, tools, settings).run("Can we ship?")

    assert result.status is RunStatus.PARTIAL
    statuses = {s.step_id: s.status for s in result.steps}
    assert statuses == {"s1": StepStatus.FAILED, "s2": StepStatus.SUCCEEDED}
    assert "Partial" in result.answer


async def test_a_failure_before_any_plan_exists_is_reported_not_raised(agents, tools, settings):
    class Broken:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("provider down")

    result = await _orchestrator(Broken(), agents, tools, settings).run("Review PR 482")

    assert result.status is RunStatus.FAILED
    assert "could not be completed" in result.answer
    assert result.trace[-1].type is EventType.RUN_FAILED


async def test_the_run_budget_is_enforced(agents, tools, settings):
    import asyncio

    class Slow:
        async def complete(self, *args, **kwargs):
            await asyncio.sleep(1)

    impatient = settings.model_copy(update={"run_timeout_seconds": 0.05})
    result = await _orchestrator(Slow(), agents, tools, impatient).run("Review PR 482")

    assert result.status is RunStatus.FAILED
    assert "budget" in result.answer


async def test_every_token_spent_is_reported_including_planning(agents, tools, settings):
    """Usage has to cover planning, not just the specialists.

    The planner's tokens were once dropped on the floor, so a run could report
    three LLM calls and the cost of one. The scripted client bills a flat
    100/20 per call, so the run total must equal what it actually served.
    """
    llm = ScriptedLLM()
    llm.when(is_first_plan, plan_reply(step("s1", "code_reviewer", "Review PR 482.")))
    llm.when(agent_turn("code_reviewer"), Reply(content="**Verdict:** request-changes"))
    llm.when(is_replan, no_followup(), sticky=True)

    result = await _orchestrator(llm, agents, tools, settings).run("Review PR 482")

    assert result.usage.llm_calls == len(llm.calls)
    assert result.usage.prompt_tokens == len(llm.calls) * 100
    assert result.usage.completion_tokens == len(llm.calls) * 20
    assert result.usage.total_tokens == len(llm.calls) * 120


async def test_a_direct_answer_still_reports_what_the_planning_cost(agents, tools, settings):
    """Answering without delegating is cheap, not free."""
    llm = ScriptedLLM()
    llm.when(is_first_plan, direct_reply("The release train runs Thursdays at 14:00 UTC."))

    result = await _orchestrator(llm, agents, tools, settings).run("When is the release train?")

    assert result.usage.llm_calls == 1
    assert result.usage.total_tokens == 120
