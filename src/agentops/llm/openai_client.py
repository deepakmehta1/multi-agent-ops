"""OpenAI-backed implementation of `LLMClient`, with bounded retries."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from agentops.core.exceptions import LLMError
from agentops.core.logging import get_logger
from agentops.llm.base import Completion, Message, ToolCall, ToolSpec, Usage

log = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAIClient:
    """Thin adapter over `AsyncOpenAI`.

    Retries connection errors, rate limits and 5xx with jittered exponential
    backoff; anything else (a bad request, a revoked key) fails straight away
    because retrying it would only burn the caller's time.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        # Retries are handled here so each attempt can be traced and logged.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self._max_retries = max_retries

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        json_object: bool = False,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_wire() for m in messages],
        }
        if tools:
            kwargs["tools"] = [t.to_wire() for t in tools]
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                return _to_completion(response)
            except (APIConnectionError, RateLimitError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code not in _RETRYABLE_STATUS:
                    raise LLMError(
                        f"OpenAI rejected the request: {exc.message}",
                        details={"status_code": exc.status_code},
                    ) from exc
                last_error = exc

            if attempt == self._max_retries:
                break
            delay = min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.25)
            log.warning(
                "llm.retry",
                extra={"attempt": attempt, "delay_s": round(delay, 2), "model": model},
            )
            await asyncio.sleep(delay)

        raise LLMError(
            f"OpenAI call failed after {self._max_retries} attempts: {last_error}",
        ) from last_error

    async def aclose(self) -> None:
        await self._client.close()


def _to_completion(response: Any) -> Completion:
    choice = response.choices[0]
    raw_calls = getattr(choice.message, "tool_calls", None) or []
    tool_calls = [
        ToolCall.from_raw(call.id, call.function.name, call.function.arguments)
        for call in raw_calls
        if getattr(call, "function", None) is not None
    ]
    usage = response.usage
    return Completion(
        message=Message(
            role="assistant",
            content=choice.message.content,
            tool_calls=tool_calls,
        ),
        usage=Usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        ),
        model=response.model,
        finish_reason=choice.finish_reason,
    )
