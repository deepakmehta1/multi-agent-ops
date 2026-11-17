"""The roster the orchestrator delegates to."""

from __future__ import annotations

from agentops.agents.base import Agent
from agentops.agents.code_reviewer import CODE_REVIEWER
from agentops.agents.incident_responder import INCIDENT_RESPONDER
from agentops.agents.release_manager import RELEASE_MANAGER
from agentops.agents.test_engineer import TEST_ENGINEER
from agentops.core.exceptions import UnknownAgentError

DEFAULT_AGENTS: tuple[Agent, ...] = (
    CODE_REVIEWER,
    TEST_ENGINEER,
    INCIDENT_RESPONDER,
    RELEASE_MANAGER,
)


class AgentRegistry:
    def __init__(self, agents: tuple[Agent, ...] | list[Agent] = DEFAULT_AGENTS) -> None:
        self._agents: dict[str, Agent] = {agent.name: agent for agent in agents}

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            raise UnknownAgentError(
                f"No agent named {name!r}.", details={"available_agents": self.names()}
            ) from None

    def names(self) -> list[str]:
        return list(self._agents)

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def describe(self) -> list[dict]:
        return [agent.describe() for agent in self._agents.values()]

    def roster_prompt(self) -> str:
        """The block of text the planner reads to choose delegates."""
        return "\n\n".join(
            f"### {agent.name} ({agent.title})\n"
            f"{agent.mandate.strip()}\n"
            f"Tools available to it: {', '.join(agent.tool_names)}"
            for agent in self._agents.values()
        )

    def __len__(self) -> int:
        return len(self._agents)
