"""Dependency wiring.

The registries are built once per process. The LLM client is resolved per
request, because a caller may supply their own key with `X-OpenAI-Key` -- that
keeps one deployment usable by several teams without sharing a billing account.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from agentops.agents.registry import AgentRegistry
from agentops.core.config import Settings, get_settings
from agentops.core.exceptions import MissingCredentialsError
from agentops.llm.base import LLMClient
from agentops.llm.openai_client import OpenAIClient
from agentops.orchestrator.orchestrator import Orchestrator
from agentops.runtime.store import RunStore
from agentops.tools.base import ToolRegistry


def get_app_settings(request: Request) -> Settings:
    """The settings for this app instance.

    Prefers the instance stored on `app.state` (set by the app factory, and overridable in
    tests) and falls back to the cached environment settings.
    """
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    return settings


def get_agent_registry(request: Request) -> AgentRegistry:
    """The shared roster of specialists, built once at startup."""
    return request.app.state.agents


def get_tool_registry(request: Request) -> ToolRegistry:
    """The shared tool registry, built once at startup."""
    return request.app.state.tools


def get_run_store(request: Request) -> RunStore:
    """The store finished runs are saved to and replayed from."""
    return request.app.state.runs


def get_llm_client(
    request: Request,
    settings: Annotated[Settings, Depends(get_app_settings)],
    x_openai_key: Annotated[str | None, Header(alias="X-OpenAI-Key")] = None,
) -> LLMClient:
    """Resolve the model provider for this request.

    Order of precedence:

    1. `app.state.llm`, if set. A test or local demo pre-sets a scripted client; no key is
       needed at all.
    2. The `X-OpenAI-Key` header. A caller's own key WINS over the server's, which is what
       lets several teams share one deployment without sharing a billing account.
    3. The server's `OPENAI_API_KEY`.

    Clients are cached per key, so repeated requests reuse one HTTP connection pool.

    Raises:
        MissingCredentialsError: None of the above supplied a key.
    """
    # Checked first so tests never touch the header or the cache.
    override: LLMClient | None = getattr(request.app.state, "llm", None)
    if override is not None:
        return override

    # Caller's key first, then the server's. An empty header counts as absent.
    api_key = x_openai_key or settings.openai_api_key
    if not api_key:
        raise MissingCredentialsError(
            "No OpenAI API key. Set OPENAI_API_KEY on the service or send one as the "
            "X-OpenAI-Key header."
        )

    # Keyed by the API key itself. The key lives only in process memory here and is never logged.
    cache: dict[str, OpenAIClient] = request.app.state.llm_cache
    client = cache.get(api_key)
    if client is None:
        client = OpenAIClient(
            api_key=api_key,
            base_url=settings.openai_base_url,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_seconds,
        )
        # Bounded so a stream of distinct caller keys cannot grow this forever.
        if len(cache) >= 32:
            # Evict the oldest entry (insertion order, so FIFO rather than LRU). The evicted client
            # is dropped without being closed; its connections are released when it is
            # garbage-collected.
            cache.pop(next(iter(cache)))
        cache[api_key] = client
    return client


def get_orchestrator(
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    agents: Annotated[AgentRegistry, Depends(get_agent_registry)],
    tools: Annotated[ToolRegistry, Depends(get_tool_registry)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> Orchestrator:
    """Build an orchestrator for this request.

    Cheap and per-request on purpose: the expensive parts (registries, clients) are shared,
    and the orchestrator holds no per-run state, so there is nothing to clean up.
    """
    return Orchestrator(llm=llm, agents=agents, tools=tools, settings=settings)


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
AgentsDep = Annotated[AgentRegistry, Depends(get_agent_registry)]
ToolsDep = Annotated[ToolRegistry, Depends(get_tool_registry)]
RunStoreDep = Annotated[RunStore, Depends(get_run_store)]
