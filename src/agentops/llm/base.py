"""Provider-agnostic chat interface.

Everything above this module talks in terms of `Message` / `ToolSpec` / `Completion`,
so swapping OpenAI for another provider (or for the scripted fake used in tests)
touches exactly one file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """A model's request to invoke one tool."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_raw(cls, call_id: str, name: str, raw_arguments: str) -> ToolCall:
        """Build a `ToolCall` from the provider's raw JSON argument string.

        Models occasionally emit malformed or non-object JSON. Raising here would abort the whole
        run over one bad turn, so junk is kept instead: unparseable text becomes
        `{"_unparsed": ...}` and a non-object value becomes `{"value": ...}`. The tool layer
        then rejects those unknown arguments and tells the model what it should have sent.

        Args:
            call_id: The provider's id for this call; the tool result must echo it back.
            name: Name of the requested tool.
            raw_arguments: The JSON text exactly as the model produced it (may be empty).
        """
        try:
            parsed = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            # Keep the junk so the tool layer can answer the model with a precise error.
            parsed = {"_unparsed": raw_arguments}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        return cls(id=call_id, name=name, arguments=parsed)


@dataclass(slots=True)
class Message:
    """One turn in a conversation, in a provider-neutral shape.

    `role` is "system", "user", "assistant" or "tool". An assistant turn may carry
    `tool_calls` instead of text; the matching "tool" turns answer them by `tool_call_id`.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Render into the OpenAI chat-completions message shape."""
        payload: dict[str, Any] = {"role": self.role}
        if self.role == "tool":
            payload["content"] = self.content or ""
            payload["tool_call_id"] = self.tool_call_id
            return payload
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
            payload.setdefault("content", None)
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(slots=True)
class ToolSpec:
    """A function the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        """Render as the `{type: function, function: {...}}` entry OpenAI expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class Usage:
    """Token counts for one or more model calls.

    Supports `+` so a run can total the planner, every agent turn and the synthesizer.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        """Sum two usages field by field, returning a new one (neither is mutated)."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(slots=True)
class Completion:
    """The result of one model call: the assistant message, its cost, and why it stopped."""

    message: Message
    usage: Usage
    model: str
    finish_reason: str | None = None

    @property
    def text(self) -> str:
        """The assistant's text, or an empty string for a pure tool-calling turn."""
        return self.message.content or ""


@runtime_checkable
class LLMClient(Protocol):
    """The single seam between this service and any model provider."""

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        json_object: bool = False,
    ) -> Completion:
        """Return one assistant turn, which may contain tool calls."""
        ...
