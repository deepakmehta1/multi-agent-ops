"""Domain exceptions. The API layer maps these onto HTTP status codes."""

from __future__ import annotations


class AgentOpsError(Exception):
    """Base class for everything this service raises deliberately."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        """Create the error.

        Args:
            message: Human-readable explanation, safe to show to the caller.
            details: Optional structured context (for example the valid values a caller
                should have used). Surfaced as `details` in the HTTP error body.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AgentOpsError):
    """The service is misconfigured (503): the caller cannot fix this."""

    status_code = 503
    code = "configuration_error"


class MissingCredentialsError(ConfigurationError):
    """No OpenAI key was available (401): neither on the server nor in `X-OpenAI-Key`."""

    status_code = 401
    code = "missing_credentials"


class LLMError(AgentOpsError):
    """The model provider failed in a way retrying will not fix (502), or retries ran out."""

    status_code = 502
    code = "llm_error"


class PlanValidationError(AgentOpsError):
    """The planner produced a plan we refuse to execute (422).

    Raised for an unknown agent, a duplicate or empty step, a dependency that does not
    exist or points at itself, a cycle, or too many steps. The planner gets exactly one
    chance to repair the plan before this surfaces.
    """

    status_code = 422
    code = "invalid_plan"


class UnknownAgentError(AgentOpsError):
    """A specialist was requested that is not on the roster (404)."""

    status_code = 404
    code = "unknown_agent"


class ToolExecutionError(AgentOpsError):
    """Raised inside a tool. Caught by the agent loop and fed back to the model."""

    status_code = 400
    code = "tool_error"


class BudgetExceededError(AgentOpsError):
    """A configured budget (steps, iterations, time) was exhausted (429)."""

    status_code = 429
    code = "budget_exceeded"


class RunNotFoundError(AgentOpsError):
    """No run with that id is stored, or it has aged out of history (404)."""

    status_code = 404
    code = "run_not_found"
