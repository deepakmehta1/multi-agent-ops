from __future__ import annotations

import json

from agentops.core.exceptions import ToolExecutionError
from agentops.tools.base import Tool, ToolRegistry, obj, string


async def test_tool_returns_json_payload(tools):
    result = await tools.get("get_pull_request").invoke({"number": 482})
    payload = json.loads(result.content)
    assert result.ok
    assert payload["title"] == "Add idempotency keys to POST /charges"
    # The diff is withheld unless explicitly requested.
    assert "patch" not in payload["changed_files"][0]


async def test_tool_includes_diff_on_request(tools):
    result = await tools.get("get_pull_request").invoke({"number": 482, "include_diff": True})
    payload = json.loads(result.content)
    assert "patch" in payload["changed_files"][0]


async def test_unknown_argument_is_reported_not_raised(tools):
    result = await tools.get("get_pull_request").invoke({"number": 482, "colour": "red"})
    assert not result.ok
    assert "colour" in result.content
    assert "accepted_arguments" in result.content


async def test_missing_required_argument_is_reported(tools):
    result = await tools.get("get_coverage_report").invoke({})
    assert not result.ok
    assert "repo" in result.content


async def test_tool_error_becomes_a_readable_result(tools):
    result = await tools.get("get_service_metrics").invoke({"service": "nope"})
    assert not result.ok
    assert "known_services" in result.content


async def test_unexpected_exception_is_contained():
    def boom() -> None:
        raise RuntimeError("kaboom")

    tool = Tool(name="boom", description="", parameters=obj({}), handler=boom)
    result = await tool.invoke({})
    assert not result.ok
    assert "RuntimeError: kaboom" in result.content


async def test_declared_tool_error_keeps_its_details():
    def fail() -> None:
        raise ToolExecutionError("nope", details={"hint": "try again"})

    tool = Tool(name="fail", description="", parameters=obj({}), handler=fail)
    result = await tool.invoke({})
    assert json.loads(result.content)["details"] == {"hint": "try again"}


def test_registry_rejects_duplicates_and_unknown_names():
    registry = ToolRegistry([Tool("a", "", obj({"x": string("x")}), lambda x=None: x)])
    assert registry.names() == ["a"]
    try:
        registry.register(Tool("a", "", obj({}), lambda: None))
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate registration should fail")

    try:
        registry.select(["a", "missing"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown tool should fail")


def test_every_agent_tool_exists(tools, agents):
    for agent in agents.all():
        assert tools.select(agent.tool_names)
