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
        """Build the user message an agent is given for this task.

        Always contains the original request (so the agent understands the bigger picture) and
        its own assignment. When upstream steps already ran, their reports are appended under
        "Findings already gathered", with an instruction not to redo that work. This is the only
        channel through which one agent learns what another found.
        """
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
    """One tool call as it is recorded for the response and the UI.

    `result_preview` is truncated; the model always sees the full result, the record does not.
    """

    name: str
    arguments: dict[str, Any]
    ok: bool
    result_preview: str
    duration_ms: int


@dataclass(slots=True)
class AgentResult:
    """What an agent returns: its report plus the bookkeeping about how it got there.

    `iterations` is how many model turns it used; `truncated` is True when it ran out of its
    tool budget and was forced to write the report from what it had.
    """

    agent: str
    output: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0
    truncated: bool = False


@dataclass(slots=True)
class Agent:
    """A specialist: a role, its tools, and the report it must produce.

    Instances are plain data (see `code_reviewer.py` and friends). All of them share the same
    `run` loop; what makes a reviewer differ from an incident responder is only these fields.
    """

    name: str
    title: str
    mandate: str
    """One or two sentences the orchestrator reads when deciding whom to delegate to."""
    instructions: str
    # The shape the report must take; appended to the system prompt.
    output_contract: str
    # Names of the tools this agent may call; enforced in `_call_tool`, not just suggested.
    tool_names: list[str]
    # Low on purpose: these agents report facts from tools, so run-to-run variation is noise.
    temperature: float = 0.1

    def describe(self) -> dict[str, Any]:
        """A JSON-friendly summary of this agent, as served by `GET /v1/agents`."""
        return {
            "name": self.name,
            "title": self.title,
            "mandate": self.mandate,
            "tools": self.tool_names,
        }

    def system_prompt(self) -> str:
        """Assemble the system prompt: role, role-specific instructions, then the shared rules.

        The shared "How to work" rules are identical for every agent and carry the behaviour we
        rely on regardless of role: ground claims in tool data, call independent tools in
        parallel, read tool errors, admit when the data does not answer the question. The
        agent's `output_contract` closes the prompt so the report has a predictable shape.
        """
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
        """Work on one task: call tools until the model stops asking, then return its report.

        The loop, per iteration:

        1. Ask the model what to do next, offering this agent's tools.
        2. If it answers without requesting a tool, that answer is the report - done.
        3. Otherwise run every tool it requested concurrently and feed the results back.

        If `max_agent_iterations` turns pass without a final answer, the tools are taken away and
        the model is told to write its report from what it knows, saying what it could not
        verify. That result is flagged `truncated` rather than treated as a failure, because a
        partial but honest report is still useful to the orchestrator.

        Args:
            task: The assignment, plus any upstream findings.
            ctx: Per-run services: model client, tools, trace bus, settings.
            step_id: The plan step this work belongs to; stamped on every trace event.

        Returns:
            The report and how it was produced.

        Raises:
            LLMError: The model provider failed (propagates to the executor, which fails the
                step without aborting the run).
        """
        # Resolve the agent's tools up front: a typo in an agent definition fails here, loudly,
        # instead of on the first request that happens to need that tool.
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
            # The assistant's turn (text and/or tool requests) must join the history BEFORE the tool
            # results below: the API rejects a tool message that has no preceding request for it.
            messages.append(completion.message)

            # No tool requested: the model has given its final answer.
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

            # Run the requested tools concurrently. gather() returns results in call order, which is
            # what lets them be paired back with their requests by position below.
            results = await asyncio.gather(
                *(self._call_tool(call, ctx, step_id) for call in completion.message.tool_calls)
            )
            # One tool message per request, in request order; `strict=True` turns a mismatch into an
            # error rather than silently mis-pairing a result with the wrong request.
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
        """Execute one tool call the model asked for and return `(record, content_for_the_model)`.

        Two jobs beyond running the tool. First, it enforces the agent's mandate: a call to a
        tool that exists but is not in `tool_names` is refused here, in code, and the refusal is
        fed back to the model like any tool error. Scoping tools by prompt alone would be a
        suggestion; this makes it a rule. Second, it emits `tool.called` / `tool.result` events
        so the call shows up live in the trace.
        """
        tool = ctx.registry.get(call.name)
        ctx.bus.emit(
            EventType.TOOL_CALLED,
            agent=self.name,
            step_id=step_id,
            message=call.name,
            arguments=call.arguments,
        )
        started = perf_counter()

        # Mandate check. The model may name a tool that does not exist OR one that exists but
        # belongs to another specialist; both get the same refusal, which lists what IS available.
        # (The list is rendered with Python repr - it is read by the model, not parsed.)
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
            # Previews are truncated to keep the trace small; the model was given the full content.
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
