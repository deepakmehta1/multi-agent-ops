"""The specialist agent: a role, a tool budget, and a bounded tool-calling loop.

Every specialist in this system is an instance of `Agent`. What distinguishes a
code reviewer from an incident responder is its prompt, the tools it is allowed
to touch, and the report it must produce -- not its code path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from agentops.core.config import Settings
from agentops.core.logging import get_logger
from agentops.llm.base import LLMClient, Message, ToolCall, Usage
from agentops.runtime.events import EventBus, EventType
from agentops.tools.base import ToolRegistry

log = get_logger(__name__)


@dataclass(slots=True)
class RunContext:
    """Everything an agent needs that is per-run rather than per-agent."""

    run_id: str
    llm: LLMClient
    registry: ToolRegistry
    bus: EventBus
    settings: Settings


@dataclass(slots=True)
class AgentTask:
    """One unit of delegated work."""

    objective: str
    request: str
    context: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        parts = [
            "## Original request from the team",
            self.request,
            "",
            "## Your assignment",
            self.objective,
        ]
        if self.context:
            parts += ["", "## Findings already gathered by other agents"]
            parts += [f"\n### From {name}\n{body}" for name, body in self.context.items()]
            parts.append(
                "\nBuild on those findings. Do not repeat work that is already done above."
            )
        return "\n".join(parts)


@dataclass(slots=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result_preview: str
    duration_ms: int


@dataclass(slots=True)
class AgentResult:
    agent: str
    output: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0
    truncated: bool = False


@dataclass(slots=True)
class Agent:
    name: str
    title: str
    mandate: str
    """One or two sentences the orchestrator reads when deciding whom to delegate to."""
    instructions: str
    output_contract: str
    tool_names: list[str]
    temperature: float = 0.1

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "mandate": self.mandate,
            "tools": self.tool_names,
        }

    def system_prompt(self) -> str:
        return (
            f"You are the {self.title} on the Northwind Payments delivery team.\n\n"
            f"{self.instructions.strip()}\n\n"
            "## How to work\n"
            "- Ground every claim in data you fetched with your tools. Never invent a pull "
            "request number, a metric, a test name or a person.\n"
            "- Call tools in parallel when the calls do not depend on each other.\n"
            "- If a tool returns an error, read it: it usually tells you the valid inputs.\n"
            "- If the data genuinely does not answer the question, say so plainly rather than "
            "filling the gap with a guess.\n"
            "- Be concise and concrete. You are writing for busy engineers.\n\n"
            "## Your report must follow this shape\n"
            f"{self.output_contract.strip()}"
        )

    async def run(self, task: AgentTask, ctx: RunContext, *, step_id: str) -> AgentResult:
        tools = ctx.registry.select(self.tool_names)
        specs = [tool.spec() for tool in tools]
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt()),
            Message(role="user", content=task.render()),
        ]

        records: list[ToolCallRecord] = []
        usage = Usage()
        max_iterations = ctx.settings.max_agent_iterations

        for iteration in range(1, max_iterations + 1):
            completion = await ctx.llm.complete(
                messages,
                model=ctx.settings.agent_model,
                tools=specs,
                temperature=self.temperature,
            )
            usage = usage + completion.usage
            messages.append(completion.message)

            if not completion.message.tool_calls:
                ctx.bus.emit(
                    EventType.STEP_COMPLETED,
                    agent=self.name,
                    step_id=step_id,
                    message=f"{self.title} finished after {iteration} iteration(s)",
                    tool_calls=len(records),
                    tokens=usage.total_tokens,
                )
                return AgentResult(
                    agent=self.name,
                    output=completion.text,
                    tool_calls=records,
                    usage=usage,
                    iterations=iteration,
                )

            ctx.bus.emit(
                EventType.AGENT_THINKING,
                agent=self.name,
                step_id=step_id,
                message=completion.text or f"Calling {len(completion.message.tool_calls)} tool(s)",
                tools=[call.name for call in completion.message.tool_calls],
            )

            results = await asyncio.gather(
                *(self._call_tool(call, ctx, step_id) for call in completion.message.tool_calls)
            )
            for call, (record, content) in zip(completion.message.tool_calls, results, strict=True):
                records.append(record)
                messages.append(
                    Message(role="tool", content=content, tool_call_id=call.id, name=call.name)
                )

        # Out of iterations: force one final answer with the tools taken away.
        messages.append(
            Message(
                role="user",
                content=(
                    "You have used your entire tool budget. Write your final report now from "
                    "what you already know, and state explicitly what you could not verify."
                ),
            )
        )
        completion = await ctx.llm.complete(
            messages, model=ctx.settings.agent_model, temperature=self.temperature
        )
        usage = usage + completion.usage
        log.warning(
            "agent.truncated",
            extra={"run_id": ctx.run_id, "agent": self.name, "iterations": max_iterations},
        )
        # A truncated step still completed, and anything watching the trace has to
        # see it close -- and see that it closed early. Without this the stream
        # shows the step starting and then nothing at all.
        ctx.bus.emit(
            EventType.STEP_COMPLETED,
            agent=self.name,
            step_id=step_id,
            message=f"{self.title} ran out of tool budget after {max_iterations} iteration(s)",
            tool_calls=len(records),
            tokens=usage.total_tokens,
            truncated=True,
        )
        return AgentResult(
            agent=self.name,
            output=completion.text,
            tool_calls=records,
            usage=usage,
            iterations=max_iterations,
            truncated=True,
        )

    async def _call_tool(
        self, call: ToolCall, ctx: RunContext, step_id: str
    ) -> tuple[ToolCallRecord, str]:
        tool = ctx.registry.get(call.name)
        ctx.bus.emit(
            EventType.TOOL_CALLED,
            agent=self.name,
            step_id=step_id,
            message=call.name,
            arguments=call.arguments,
        )
        started = perf_counter()

        if tool is None or call.name not in self.tool_names:
            content = (
                f'{{"error": "Tool {call.name!r} is not available to you.", '
                f'"available_tools": {self.tool_names}}}'
            )
            ok = False
        else:
            result = await tool.invoke(call.arguments)
            content = result.content
            ok = result.ok

        duration_ms = int((perf_counter() - started) * 1000)
        ctx.bus.emit(
            EventType.TOOL_RESULT,
            agent=self.name,
            step_id=step_id,
            message=call.name,
            ok=ok,
            duration_ms=duration_ms,
            preview=content[:300],
        )
        return (
            ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=ok,
                result_preview=content[:500],
                duration_ms=duration_ms,
            ),
            content,
        )
