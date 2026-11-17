"""Domain exceptions. The API layer maps these onto HTTP status codes."""

from __future__ import annotations


class AgentOpsError(Exception):
    """Base class for everything this service raises deliberately."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AgentOpsError):
    status_code = 503
    code = "configuration_error"


class MissingCredentialsError(ConfigurationError):
    status_code = 401
    code = "missing_credentials"


class LLMError(AgentOpsError):
    status_code = 502
    code = "llm_error"


class PlanValidationError(AgentOpsError):
    status_code = 422
    code = "invalid_plan"


class UnknownAgentError(AgentOpsError):
    status_code = 404
    code = "unknown_agent"


class ToolExecutionError(AgentOpsError):
    """Raised inside a tool. Caught by the agent loop and fed back to the model."""

    status_code = 400
    code = "tool_error"


class BudgetExceededError(AgentOpsError):
    status_code = 429
    code = "budget_exceeded"


class RunNotFoundError(AgentOpsError):
    status_code = 404
    code = "run_not_found"
