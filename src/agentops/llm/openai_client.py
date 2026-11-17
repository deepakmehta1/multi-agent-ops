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

# HTTP statuses worth retrying: request timeout, conflict, rate limit, and the transient
# 5xx family. Other 4xx mean the request itself is wrong and will fail again identically.
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
        """Args:
        api_key: OpenAI (or compatible) key used for every call made through this client.
        base_url: Override the API endpoint (Azure OpenAI, a gateway, a local server).
        max_retries: Total attempts per call; see the class docstring.
        timeout: Per-attempt timeout in seconds, enforced by the SDK.
        """
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
        """Make one chat-completions call, retrying transient failures.

        See `LLMClient.complete` for the arguments. Backoff between attempts is
        `min(8, 0.5 * 2**(attempt-1))` seconds (0.5, 1, 2, 4, 8, ...) plus up to 0.25 s of
        random jitter, so many concurrent callers hitting a rate limit do not retry in lockstep.

        Raises:
            LLMError: Immediately for non-retryable HTTP statuses (400, 401, 404, ...), or after
                the final attempt for connection errors, rate limits and retryable statuses.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_wire() for m in messages],
        }
        # Only advertise tools when there are some: with `tools` absent the model can only answer
        # in text, which is how an out-of-budget agent is forced to write its report.
        if tools:
            kwargs["tools"] = [t.to_wire() for t in tools]
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}

        # `kwargs` above is built once and reused for every attempt, so a retry sends exactly the
        # same request. `last_error` remembers why we gave up, for the final error message.
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                return _to_completion(response)
            # Always transient. (RateLimitError is also an APIStatusError; catching it first keeps
            # it out of the status-code check below.)
            except (APIConnectionError, RateLimitError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code not in _RETRYABLE_STATUS:
                    raise LLMError(
                        f"OpenAI rejected the request: {exc.message}",
                        details={"status_code": exc.status_code},
                    ) from exc
                last_error = exc

            # No point sleeping after the last attempt; fall through to the final error.
            if attempt == self._max_retries:
                break
            # Capped exponential backoff plus jitter.
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
        """Close the underlying HTTP client; call once when discarding this client."""
        await self._client.close()


def _to_completion(response: Any) -> Completion:
    """Convert an SDK chat-completion response into our provider-neutral `Completion`.

    Only the first choice is used (we never ask for more). Usage fields can be missing or
    None on some compatible servers, so they default to zero rather than raising.
    """
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
