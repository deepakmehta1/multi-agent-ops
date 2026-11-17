"""Shared fixtures. Every test runs against the scripted LLM -- no network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentops.agents.registry import AgentRegistry
from agentops.api.deps import get_llm_client
from agentops.core.config import Settings
from agentops.llm.fake import Reply, ScriptedLLM
from agentops.main import create_app
from agentops.runtime.events import EventBus
from agentops.tools.catalog import build_registry
from helpers import is_first_plan, is_replan, is_synthesis, no_followup, plan_reply, step


@pytest.fixture
def settings() -> Settings:
    """Settings with small limits so tests exercise the guardrails quickly.

    Notably 4 plan steps, 3 agent iterations and short timeouts, so budget and timeout paths
    are reachable without long waits.
    """
    return Settings(
        openai_api_key=None,
        orchestrator_model="test-orchestrator",
        agent_model="test-agent",
        max_plan_steps=4,
        max_replan_rounds=1,
        max_agent_iterations=3,
        step_timeout_seconds=5.0,
        run_timeout_seconds=20.0,
        max_parallel_steps=4,
        log_format="text",
        environment="test",
    )


@pytest.fixture
def agents() -> AgentRegistry:
    """The standard four-specialist roster."""
    return AgentRegistry()


@pytest.fixture
def tools():
    """The full tool registry over the seeded data."""
    return build_registry()


@pytest.fixture
def bus() -> EventBus:
    """A fresh trace bus for one run."""
    return EventBus("run_test")


@pytest.fixture
def scripted() -> ScriptedLLM:
    """A default script: one review step, no follow-up, a canned synthesis."""
    llm = ScriptedLLM(default=Reply(content="specialist report"))
    llm.when(
        is_first_plan,
        plan_reply(step("s1", "code_reviewer", "Review PR 482 on checkout-api.")),
    )
    llm.when(is_replan, no_followup(), sticky=True)
    llm.when(is_synthesis, Reply(content="Final answer."), sticky=True)
    return llm


@pytest.fixture
def client(settings: Settings, scripted: ScriptedLLM):
    """A TestClient over the real app with the scripted model swapped in.

    Overrides `get_llm_client` rather than setting a key, so the tests also cover the
    dependency wiring, and clears the override afterwards.
    """
    app = create_app(settings)
    app.dependency_overrides[get_llm_client] = lambda: scripted
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
