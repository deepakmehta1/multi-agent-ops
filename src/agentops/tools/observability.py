"""Production observability tools: deploys, metrics, logs, incidents, on-call."""

from __future__ import annotations

from agentops.systems.store import SystemsStore
from agentops.tools.base import Tool, integer, obj, string


def observability_tools(store: SystemsStore) -> list[Tool]:
    return [
        Tool(
            name="get_service_info",
            description=(
                "Ownership, tier, SLO targets and runbook link for one service. "
                "Call list_services first if you are not sure of the name."
            ),
            parameters=obj({"service": string("Service name.")}, required=["service"]),
            handler=lambda service: store.service(service),
            tags=["observability", "incident", "release"],
        ),
        Tool(
            name="list_services",
            description="Every service in the delivery org, with its owning team and repository.",
            parameters=obj({}),
            handler=store.services,
            tags=["observability", "incident", "release", "review"],
        ),
        Tool(
            name="list_deployments",
            description=(
                "Recent production deployments, newest first: version, commit, who shipped it "
                "and when. The first thing to check when something broke."
            ),
            parameters=obj(
                {
                    "service": string("Restrict to one service."),
                    "limit": integer("How many to return (default 10)."),
                }
            ),
            handler=lambda service=None, limit=10: store.deployments(
                service=service, limit=int(limit)
            ),
            tags=["observability", "incident", "release"],
        ),
        Tool(
            name="get_service_metrics",
            description=(
                "Time series for a service: error_rate_percent, p99_latency_ms and "
                "service-specific gauges such as db_connection_pool_saturation_percent or "
                "queue_depth. Omit `metric` to get every series the service reports."
            ),
            parameters=obj(
                {
                    "service": string("Service name."),
                    "metric": string("One metric name; omit for all of them."),
                },
                required=["service"],
            ),
            handler=lambda service, metric=None: store.metrics(service, metric=metric),
            tags=["observability", "incident"],
        ),
        Tool(
            name="search_logs",
            description=(
                "Search aggregated application logs. Each entry carries a count, so a single "
                "line can represent thousands of occurrences."
            ),
            parameters=obj(
                {
                    "service": string("Restrict to one service."),
                    "level": string("Log level.", enum=["DEBUG", "INFO", "WARNING", "ERROR"]),
                    "contains": string("Case-insensitive substring to match in the message."),
                    "limit": integer("Maximum entries to return (default 20)."),
                }
            ),
            handler=lambda service=None, level=None, contains=None, limit=20: store.logs(
                service=service, level=level, contains=contains, limit=int(limit)
            ),
            tags=["observability", "incident"],
        ),
        Tool(
            name="list_incidents",
            description="Incidents with severity, status, summary and timeline.",
            parameters=obj({"status": string("Filter by status.", enum=["open", "resolved"])}),
            handler=lambda status=None: store.incidents(status=status),
            tags=["observability", "incident", "release"],
        ),
        Tool(
            name="get_oncall",
            description="Who is on call right now, per team, primary and secondary.",
            parameters=obj({"team": string("Restrict to one team.")}),
            handler=lambda team=None: store.oncall(team=team),
            tags=["observability", "incident"],
        ),
    ]
