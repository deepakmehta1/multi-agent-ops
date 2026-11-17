"""Application settings, loaded from the environment (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every tunable of the service, read from environment variables and an optional `.env`.

    Precedence is the usual one for pydantic-settings: real environment variables beat
    `.env`, which beats the defaults below. Field names map to upper-case variables
    (`max_plan_steps` <- `MAX_PLAN_STEPS`). Unknown variables are ignored on purpose, so a
    shared `.env` or a noisy CI environment cannot stop the service from starting.

    The "guardrail" fields bound how much work and money one request can consume. Each of
    them is enforced in code, not left to the model's good behaviour.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ----------------------------------------------------
    openai_api_key: str | None = Field(
        default=None,
        description="Server-side key. Callers may override it per request with X-OpenAI-Key.",
    )
    # Point the OpenAI client at Azure OpenAI, a gateway, or any compatible local server.
    openai_base_url: str | None = None
    # The orchestrator plans and writes the final answer, so it gets the stronger model.
    # Specialists below only run focused tool loops and can use a cheaper one.
    orchestrator_model: str = "gpt-4.1"
    agent_model: str = "gpt-4.1-mini"
    # Retries are done by our own client (so each one is logged), not by the OpenAI SDK.
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # --- Runtime guardrails ----------------------------------------------
    # Most delegations one plan may contain. A plan above this is rejected and the planner is
    # asked once to fix it, which stops a runaway plan from fanning out into a large bill.
    max_plan_steps: int = 6
    # How many times the orchestrator may look at the results and schedule follow-up work.
    # 0 disables the review round entirely; every extra round costs another planning call.
    max_replan_rounds: int = 1
    # Tool-calling turns one specialist gets. When it is spent the agent is forced to write
    # its report from what it already knows (flagged `truncated`) rather than loop forever.
    max_agent_iterations: int = 6
    # Wall-clock limit for one delegation. A step that exceeds it fails and its dependants
    # are skipped; the rest of the run carries on.
    step_timeout_seconds: float = 90.0
    # Wall-clock limit for the whole run, planning and synthesis included. Always larger than
    # the step timeout, since steps can run in several waves.
    run_timeout_seconds: float = 300.0
    # Upper bound on delegations running at the same moment (a semaphore in the executor),
    # to keep us inside the provider's rate limits.
    max_parallel_steps: int = 4
    # Finished runs kept in memory for replay. Oldest are dropped first.
    run_history_size: int = 200

    # --- Service ----------------------------------------------------------
    environment: str = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    # Permissive by default because the demo has no cookies or credentials to protect;
    # restrict it to your own origins before exposing the service.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def has_server_key(self) -> bool:
        """True when the service was started with its own OpenAI key.

        When False, every request must carry its own key in the `X-OpenAI-Key` header;
        `/readyz` reports which of the two modes the service is in.
        """
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance. Call `get_settings.cache_clear()` in tests."""
    return Settings()
