"""Issue-tracker and release-management tools."""

from __future__ import annotations

from agentops.systems.store import SystemsStore
from agentops.tools.base import Tool, obj, string


def tracker_tools(store: SystemsStore) -> list[Tool]:
    """Tools over the issue tracker and the release gate (checklist and schedule)."""
    return [
        Tool(
            name="list_tickets",
            description="Delivery tickets with status, assignee, priority and linked pull request.",
            parameters=obj(
                {
                    "team": string("Restrict to one team."),
                    "status": string("Filter by status, e.g. 'In Review'."),
                }
            ),
            handler=lambda team=None, status=None: store.tickets(team=team, status=status),
            tags=["tracker", "release", "review", "incident"],
        ),
        Tool(
            name="get_ticket",
            description="One ticket by id, e.g. 'DEL-1123'.",
            parameters=obj({"ticket_id": string("Ticket id.")}, required=["ticket_id"]),
            handler=lambda ticket_id: store.ticket(ticket_id),
            tags=["tracker", "release", "review"],
        ),
        Tool(
            name="get_release_checklist",
            description=(
                "The gate for the next release: target date and every checklist item with its "
                "status (done / pending / blocked) and a note explaining blockers."
            ),
            parameters=obj({}),
            handler=store.release_checklist,
            tags=["tracker", "release"],
        ),
        Tool(
            name="get_organisation_context",
            description=(
                "Company-level facts: delivery teams and the release train schedule "
                "(when the freeze and the deploy window fall)."
            ),
            parameters=obj({}),
            handler=store.organisation,
            tags=["tracker", "release", "incident"],
        ),
    ]
