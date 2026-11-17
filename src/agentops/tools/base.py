"""Tool definitions and the registry agents resolve calls against."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentops.core.exceptions import ToolExecutionError
from agentops.llm.base import ToolSpec


@dataclass(slots=True)
class ToolResult:
    """What a tool hands back to the model, plus what the trace records.

    `content` is the exact text placed in the conversation; `raw` keeps the structured value
    for logging and tests; `ok` is False for errors so the trace can mark them.
    """

    content: str
    ok: bool = True
    raw: Any = None

    @classmethod
    def json(cls, payload: Any) -> ToolResult:
        """Wrap any JSON-serialisable value as a successful result."""
        return cls(content=json.dumps(payload, default=str, indent=None), raw=payload)

    @classmethod
    def error(cls, message: str, details: dict | None = None) -> ToolResult:
        """Build a failed result in the shape `{"error": ..., "details": {...}}`.

        Args:
            message: What went wrong, written for the model to read and act on.
            details: Optional extra context. Put the valid choices here (for example the known
                service names) so the model can retry correctly.
        """
        body = {"error": message}
        if details:
            # `body` is inferred as dict[str, str]; details is a dict, hence the ignore.
            body["details"] = details  # type: ignore[assignment]
        return cls(content=json.dumps(body, default=str), ok=False, raw=body)


@dataclass(slots=True)
class Tool:
    """One function an agent may call: its name, the JSON schema shown to the model, and the code.

    `description` and `parameters` are prompt text as much as metadata - the model chooses
    and fills in tools from them alone, so vague wording here causes wrong calls.
    `handler` may be sync or async; its keyword arguments are the schema's properties.
    `tags` group tools by domain for introspection.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    tags: list[str] = field(default_factory=list)

    def spec(self) -> ToolSpec:
        """The provider-neutral `ToolSpec` advertised to the model."""
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    async def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool and always return a result, never raise.

        A tool error is information, not a crash: the model is told what went wrong and usually
        recovers by calling the tool differently. In order, this:

        1. rejects arguments the schema does not list (providers do not enforce
           `additionalProperties: false` for us) and names the accepted ones;
        2. rejects missing required arguments;
        3. calls the handler, awaiting it if it returned an awaitable;
        4. converts `ToolExecutionError` into a result carrying its `details`, and any other
           exception into a plain error result, so one buggy tool cannot abort a run.

        Cancellation is the exception: `CancelledError` is re-raised so a stopped or timed-out
        run actually stops instead of being swallowed as a tool failure.
        """
        allowed = set(self.parameters.get("properties", {}))
        unknown = set(arguments) - allowed
        if unknown:
            return ToolResult.error(
                f"Unknown argument(s): {', '.join(sorted(unknown))}.",
                {"accepted_arguments": sorted(allowed)},
            )
        missing = set(self.parameters.get("required", [])) - set(arguments)
        if missing:
            return ToolResult.error(f"Missing required argument(s): {', '.join(sorted(missing))}.")

        try:
            result = self.handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
        except ToolExecutionError as exc:
            return ToolResult.error(exc.message, exc.details)
        # Never swallow cancellation (it is a BaseException on 3.8+, but be explicit): a cancelled
        # run must unwind, not be reported to the model as a tool error.
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

        # Handlers may return a ready-made ToolResult (to control `ok`); anything else is wrapped.
        if isinstance(result, ToolResult):
            return result
        return ToolResult.json(result)


class ToolRegistry:
    """Named collection of tools, sliced per agent by tag."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        """Create a registry, optionally pre-loaded with `tools`."""
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool.

        Raises:
            ValueError: A tool with this name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool called `name`, or None. Callers decide how to treat a miss."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """All tool names, sorted so listings are stable."""
        return sorted(self._tools)

    def select(self, names: list[str]) -> list[Tool]:
        """Return the named tools in the order given.

        Raises:
            ValueError: Any name is not registered. Failing here, at wiring time, catches a typo
                in an agent definition before a user ever hits it.
        """
        missing = [name for name in names if name not in self._tools]
        if missing:
            raise ValueError(f"Unknown tool(s): {', '.join(missing)}")
        return [self._tools[name] for name in names]

    def by_tag(self, tag: str) -> list[Tool]:
        """All tools carrying `tag`, in registration order."""
        return [tool for tool in self._tools.values() if tag in tool.tags]

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)


# --- small JSON Schema helpers -------------------------------------------


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Build a JSON-schema object for a tool's parameters.

    `additionalProperties` is False so the schema documents exactly what is accepted;
    `Tool.invoke` enforces it, since the provider will not.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def string(description: str, enum: list[str] | None = None) -> dict[str, Any]:
    """A string parameter, optionally restricted to the values in `enum`."""
    schema: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        schema["enum"] = enum
    return schema


def integer(description: str) -> dict[str, Any]:
    """An integer parameter."""
    return {"type": "integer", "description": description}


def boolean(description: str) -> dict[str, Any]:
    """A boolean parameter."""
    return {"type": "boolean", "description": description}
