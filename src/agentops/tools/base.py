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
    """What a tool hands back to the model, plus what the trace records."""

    content: str
    ok: bool = True
    raw: Any = None

    @classmethod
    def json(cls, payload: Any) -> ToolResult:
        return cls(content=json.dumps(payload, default=str, indent=None), raw=payload)

    @classmethod
    def error(cls, message: str, details: dict | None = None) -> ToolResult:
        body = {"error": message}
        if details:
            body["details"] = details  # type: ignore[assignment]
        return cls(content=json.dumps(body, default=str), ok=False, raw=body)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    tags: list[str] = field(default_factory=list)

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    async def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool, turning any failure into a result the model can read.

        A tool error is information, not a crash: the model is told what went
        wrong and usually recovers by calling the tool differently.
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
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

        if isinstance(result, ToolResult):
            return result
        return ToolResult.json(result)


class ToolRegistry:
    """Named collection of tools, sliced per agent by tag."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def select(self, names: list[str]) -> list[Tool]:
        missing = [name for name in names if name not in self._tools]
        if missing:
            raise ValueError(f"Unknown tool(s): {', '.join(missing)}")
        return [self._tools[name] for name in names]

    def by_tag(self, tag: str) -> list[Tool]:
        return [tool for tool in self._tools.values() if tag in tool.tags]

    def __len__(self) -> int:
        return len(self._tools)


# --- small JSON Schema helpers -------------------------------------------


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def string(description: str, enum: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        schema["enum"] = enum
    return schema


def integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}
