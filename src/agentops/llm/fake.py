"""A scripted `LLMClient` used by the tests and the offline demo.

`ScriptedLLM` answers by matching the last user/tool turn against rules, so a
test can describe a whole multi-agent conversation without a network call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentops.llm.base import Completion, Message, ToolCall, ToolSpec, Usage


@dataclass
class Reply:
    """One canned assistant turn."""

    content: str | None = None
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def to_message(self, seq: int) -> Message:
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=[
                ToolCall(id=f"call_{seq}_{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(self.tool_calls)
            ],
        )


Matcher = Callable[[list[Message], list[ToolSpec]], bool]


class ScriptedLLM:
    """Deterministic stand-in for a real provider.

    Rules are `(matcher, reply)` pairs evaluated in order; the first match wins.
    A rule fires at most once unless `sticky=True`, which keeps conversations
    from looping forever on the same canned answer.
    """

    def __init__(self, default: Reply | None = None) -> None:
        self._rules: list[tuple[Matcher, Reply, bool, list[int]]] = []
        self._default = default or Reply(content="ok")
        self.calls: list[dict[str, Any]] = []
        self._seq = 0

    def when(
        self,
        matcher: Matcher | str,
        reply: Reply,
        *,
        sticky: bool = False,
    ) -> ScriptedLLM:
        """Register a rule. A string matcher means 'this substring appears anywhere'."""
        if isinstance(matcher, str):
            needle = matcher

            def matcher_fn(messages: list[Message], _tools: list[ToolSpec]) -> bool:
                return any(needle in (m.content or "") for m in messages)

            resolved: Matcher = matcher_fn
        else:
            resolved = matcher
        self._rules.append((resolved, reply, sticky, []))
        return self

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        json_object: bool = False,
    ) -> Completion:
        tools = tools or []
        self._seq += 1
        self.calls.append(
            {
                "model": model,
                "messages": [m.to_wire() for m in messages],
                "tools": [t.name for t in tools],
                "json_object": json_object,
            }
        )

        reply = self._default
        for matcher, candidate, sticky, used in self._rules:
            if used and not sticky:
                continue
            if matcher(messages, tools):
                used.append(self._seq)
                reply = candidate
                break

        if json_object and reply.content:
            json.loads(reply.content)  # fail loudly on a malformed fixture

        return Completion(
            message=reply.to_message(self._seq),
            usage=Usage(prompt_tokens=100, completion_tokens=20),
            model=model,
            finish_reason="tool_calls" if reply.tool_calls else "stop",
        )
