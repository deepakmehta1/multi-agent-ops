"""Matchers and reply builders for driving the scripted LLM in tests."""

from __future__ import annotations

import json
from typing import Any

from agentops.agents.registry import AgentRegistry
from agentops.llm.base import Message, ToolSpec
from agentops.llm.fake import Reply


def system_prompt(messages: list[Message]) -> str:
    return messages[0].content or ""


def is_planner(messages: list[Message], tools: list[ToolSpec]) -> bool:
    return not tools and "Reply with JSON only" in system_prompt(messages)


def is_replan(messages: list[Message], tools: list[ToolSpec]) -> bool:
    return is_planner(messages, tools) and any(
        "Is anything still missing" in (m.content or "") for m in messages
    )


def is_first_plan(messages: list[Message], tools: list[ToolSpec]) -> bool:
    return is_planner(messages, tools) and not is_replan(messages, tools)


def is_synthesis(messages: list[Message], tools: list[ToolSpec]) -> bool:
    return not tools and "you answer the person who asked" in system_prompt(messages)


def agent_turn(agent_name: str):
    """Match a call made by one specific specialist."""
    title = AgentRegistry().get(agent_name).title

    def matcher(messages: list[Message], _tools: list[ToolSpec]) -> bool:
        return f"You are the {title}" in system_prompt(messages)

    return matcher


def after_tool_result(messages: list[Message], _tools: list[ToolSpec]) -> bool:
    return any(message.role == "tool" for message in messages)


def step(step_id: str, agent: str, objective: str, depends_on: list[str] | None = None) -> dict:
    return {
        "id": step_id,
        "agent": agent,
        "objective": objective,
        "depends_on": depends_on or [],
    }


def plan_reply(*steps: dict[str, Any], reasoning: str = "test plan") -> Reply:
    return Reply(
        content=json.dumps({"reasoning": reasoning, "answer_directly": None, "steps": list(steps)})
    )


def direct_reply(answer: str) -> Reply:
    return Reply(
        content=json.dumps(
            {"reasoning": "no delegation needed", "answer_directly": answer, "steps": []}
        )
    )


def no_followup() -> Reply:
    return Reply(content=json.dumps({"reasoning": "complete", "steps": []}))
