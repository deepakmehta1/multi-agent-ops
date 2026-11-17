"""Application settings, loaded from the environment (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    openai_base_url: str | None = None
    orchestrator_model: str = "gpt-4.1"
    agent_model: str = "gpt-4.1-mini"
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # --- Runtime guardrails ----------------------------------------------
    max_plan_steps: int = 6
    max_replan_rounds: int = 1
    max_agent_iterations: int = 6
    step_timeout_seconds: float = 90.0
    run_timeout_seconds: float = 300.0
    max_parallel_steps: int = 4
    run_history_size: int = 200

    # --- Service ----------------------------------------------------------
    environment: str = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def has_server_key(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance. Call `get_settings.cache_clear()` in tests."""
    return Settings()
