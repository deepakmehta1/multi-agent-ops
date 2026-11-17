"""Matchers and reply builders for driving the scripted LLM in tests."""

from __future__ import annotations

import json
from typing import Any

from agentops.agents.registry import AgentRegistry
from agentops.llm.base import Message, ToolSpec
from agentops.llm.fake import Reply


def system_prompt(messages: list[Message]) -> str:
    """The system prompt of a conversation (always its first message)."""
    return messages[0].content or ""


def is_planner(messages: list[Message], tools: list[ToolSpec]) -> bool:
    """True for a planning call: JSON mode, no tools offered, and the planner's prompt."""
    return not tools and "Reply with JSON only" in system_prompt(messages)


def is_replan(messages: list[Message], tools: list[ToolSpec]) -> bool:
    """True for the planner's follow-up call, identified by the review question it asks."""
    return is_planner(messages, tools) and any(
        "Is anything still missing" in (m.content or "") for m in messages
    )


def is_first_plan(messages: list[Message], tools: list[ToolSpec]) -> bool:
    """True for the initial planning call (a planning call that is not a replan)."""
    return is_planner(messages, tools) and not is_replan(messages, tools)


def is_synthesis(messages: list[Message], tools: list[ToolSpec]) -> bool:
    """True for the final-answer call, identified by the synthesizer's prompt."""
    return not tools and "you answer the person who asked" in system_prompt(messages)


def agent_turn(agent_name: str):
    """Match a call made by one specific specialist."""
    title = AgentRegistry().get(agent_name).title

    def matcher(messages: list[Message], _tools: list[ToolSpec]) -> bool:
        """True when the conversation's system prompt is this specialist's."""
        return f"You are the {title}" in system_prompt(messages)

    return matcher


def after_tool_result(messages: list[Message], _tools: list[ToolSpec]) -> bool:
    """True once the conversation contains a tool result, i.e. after tools ran."""
    return any(message.role == "tool" for message in messages)


def step(step_id: str, agent: str, objective: str, depends_on: list[str] | None = None) -> dict:
    """Build a plan-step dict as the planner would emit it."""
    return {
        "id": step_id,
        "agent": agent,
        "objective": objective,
        "depends_on": depends_on or [],
    }


def plan_reply(*steps: dict[str, Any], reasoning: str = "test plan") -> Reply:
    """A scripted planner reply containing `steps`, ready to register with `when`."""
    return Reply(
        content=json.dumps({"reasoning": reasoning, "answer_directly": None, "steps": list(steps)})
    )


def direct_reply(answer: str) -> Reply:
    """A scripted planner reply that answers directly and delegates nothing."""
    return Reply(
        content=json.dumps(
            {"reasoning": "no delegation needed", "answer_directly": answer, "steps": []}
        )
    )


def no_followup() -> Reply:
    """A scripted replan reply meaning "nothing more to do" (an empty step list)."""
    return Reply(content=json.dumps({"reasoning": "complete", "steps": []}))
