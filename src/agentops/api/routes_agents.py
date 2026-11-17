"""Introspection: who is on the roster and what they can touch."""

from __future__ import annotations

from fastapi import APIRouter

from agentops.api.deps import AgentsDep, SettingsDep, ToolsDep
from agentops.schemas import AgentsResponse, AgentView, ToolView

router = APIRouter(prefix="/v1", tags=["introspection"])


@router.get("/agents", response_model=AgentsResponse, summary="List the agent roster")
async def list_agents(agents: AgentsDep, settings: SettingsDep) -> AgentsResponse:
    """The orchestrator's role and limits, plus each specialist's mandate and tools.

    This is the same roster text the planner reads when deciding whom to delegate to.
    """
    return AgentsResponse(
        orchestrator={
            "role": "Delivery Lead",
            "model": settings.orchestrator_model,
            "responsibilities": [
                "break a request into a dependency graph of delegations",
                "run independent steps in parallel",
                "review the results and schedule follow-up work",
                "synthesize one answer from the specialists' reports",
            ],
            "max_plan_steps": settings.max_plan_steps,
            "max_replan_rounds": settings.max_replan_rounds,
        },
        agents=[
            AgentView(
                name=agent.name,
                title=agent.title,
                mandate=agent.mandate.strip(),
                tools=agent.tool_names,
            )
            for agent in agents.all()
        ],
    )


@router.get("/tools", response_model=list[ToolView], summary="List the tool catalogue")
async def list_tools(tools: ToolsDep, agents: AgentsDep) -> list[ToolView]:
    """Every tool in the catalogue with its JSON schema and which agents may call it.

    `used_by` makes the access rules visible: a tool is only reachable by the agents listed.
    """
    return [
        ToolView(
            name=name,
            description=tool.description,
            parameters=tool.parameters,
            used_by=[a.name for a in agents.all() if name in a.tool_names],
        )
        for name in tools.names()
        if (tool := tools.get(name)) is not None
    ]
