from __future__ import annotations

from agentops.agents.base import AgentTask, RunContext
from agentops.llm.fake import Reply, ScriptedLLM
from agentops.runtime.events import EventType
from helpers import after_tool_result, agent_turn


def _ctx(llm, tools, bus, settings) -> RunContext:
    return RunContext(run_id="run_test", llm=llm, registry=tools, bus=bus, settings=settings)


async def test_agent_calls_tools_then_reports(agents, tools, bus, settings):
    llm = ScriptedLLM()
    llm.when(
        agent_turn("code_reviewer"),
        Reply(tool_calls=[("get_pull_request", {"number": 482, "include_diff": True})]),
    )
    llm.when(
        after_tool_result,
        Reply(content="**Verdict:** request-changes"),
        sticky=True,
    )

    result = await agents.get("code_reviewer").run(
        AgentTask(objective="Review PR 482.", request="Review PR 482."),
        _ctx(llm, tools, bus, settings),
        step_id="s1",
    )

    assert result.output.startswith("**Verdict:**")
    assert [call.name for call in result.tool_calls] == ["get_pull_request"]
    assert result.tool_calls[0].ok
    assert result.iterations == 2
    assert not result.truncated
    assert result.usage.total_tokens > 0


async def test_agent_calls_independent_tools_in_one_turn(agents, tools, bus, settings):
    llm = ScriptedLLM()
    llm.when(
        agent_turn("test_engineer"),
        Reply(
            tool_calls=[
                ("get_test_suite", {"repo": "checkout-api"}),
                ("get_coverage_report", {"repo": "checkout-api"}),
            ]
        ),
    )
    llm.when(
        after_tool_result,
        Reply(content="**Coverage assessment:** gaps"),
        sticky=True,
    )

    result = await agents.get("test_engineer").run(
        AgentTask(objective="Assess coverage.", request="Assess coverage."),
        _ctx(llm, tools, bus, settings),
        step_id="s1",
    )

    assert len(result.tool_calls) == 2
    assert all(call.ok for call in result.tool_calls)


async def test_agent_cannot_reach_a_tool_outside_its_mandate(agents, tools, bus, settings):
    """The release manager has no access to the diff, even if the model asks for it."""
    llm = ScriptedLLM()
    llm.when(
        agent_turn("release_manager"),
        Reply(tool_calls=[("search_logs", {"service": "checkout-api"})]),
    )
    llm.when(
        after_tool_result,
        Reply(content="done"),
        sticky=True,
    )

    result = await agents.get("release_manager").run(
        AgentTask(objective="Check readiness.", request="Check readiness."),
        _ctx(llm, tools, bus, settings),
        step_id="s1",
    )

    assert not result.tool_calls[0].ok
    assert "not available to you" in result.tool_calls[0].result_preview


async def test_agent_is_forced_to_report_when_the_budget_runs_out(agents, tools, bus, settings):
    llm = ScriptedLLM()
    # Always asks for another tool, so the loop can only end by exhausting the budget.
    llm.when(
        lambda _m, tool_specs: bool(tool_specs),
        Reply(tool_calls=[("list_incidents", {})]),
        sticky=True,
    )
    llm.when(
        lambda _m, tool_specs: not tool_specs,
        Reply(content="partial findings"),
        sticky=True,
    )

    result = await agents.get("incident_responder").run(
        AgentTask(objective="Investigate.", request="Investigate."),
        _ctx(llm, tools, bus, settings),
        step_id="s1",
    )

    assert result.truncated
    assert result.iterations == settings.max_agent_iterations
    assert result.output == "partial findings"

    # The step has to close on the trace too, flagged as having run out of budget.
    # It used to emit nothing at all, so a stream showed the step start and stall.
    completed = [e for e in bus.history if e.type is EventType.STEP_COMPLETED]
    assert len(completed) == 1
    assert completed[0].data["truncated"] is True
    assert "tool budget" in (completed[0].message or "")


async def test_upstream_findings_are_handed_to_the_next_agent():
    task = AgentTask(
        objective="Draft the notes.",
        request="Can we ship?",
        context={"incident_responder": "INC-204 is still open."},
    )
    rendered = task.render()
    assert "INC-204 is still open." in rendered
    assert "From incident_responder" in rendered
    assert "Do not repeat work" in rendered


async def test_every_tool_call_is_traced(agents, tools, bus, settings):
    llm = ScriptedLLM()
    llm.when(agent_turn("code_reviewer"), Reply(tool_calls=[("list_pull_requests", {})]))
    llm.when(
        after_tool_result,
        Reply(content="ok"),
        sticky=True,
    )

    await agents.get("code_reviewer").run(
        AgentTask(objective="List PRs.", request="List PRs."),
        _ctx(llm, tools, bus, settings),
        step_id="s1",
    )

    kinds = [event.type for event in bus.history]
    assert EventType.TOOL_CALLED in kinds
    assert EventType.TOOL_RESULT in kinds
    assert all(event.step_id == "s1" for event in bus.history)
