"""Stand-in for the systems a delivery team actually queries.

In a real deployment each method here would call GitHub, the CI provider, the
metrics backend and the issue tracker. Keeping them behind one class means the
agents and their tools are written exactly as they would be in production --
only this file changes when the integrations become real.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentops.core.exceptions import ToolExecutionError

_SEED_PATH = Path(__file__).parent / "data" / "seed.json"


class SystemsStore:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # -- organisation ------------------------------------------------------

    def organisation(self) -> dict[str, Any]:
        return deepcopy(self._data["organisation"])

    def services(self) -> list[dict[str, Any]]:
        return deepcopy(self._data["services"])

    def service(self, name: str) -> dict[str, Any]:
        for service in self._data["services"]:
            if service["name"] == name:
                return deepcopy(service)
        raise ToolExecutionError(
            f"Unknown service {name!r}.",
            details={"known_services": [s["name"] for s in self._data["services"]]},
        )

    # -- version control ---------------------------------------------------

    def pull_requests(self, *, repo: str | None = None, state: str | None = None) -> list[dict]:
        items = self._data["pull_requests"]
        if repo:
            items = [pr for pr in items if pr["repo"] == repo or pr["repo"].endswith(f"/{repo}")]
        if state:
            items = [pr for pr in items if pr["state"] == state]
        return [{k: v for k, v in deepcopy(pr).items() if k != "changed_files"} for pr in items]

    def pull_request(self, number: int, *, include_diff: bool = False) -> dict[str, Any]:
        for pr in self._data["pull_requests"]:
            if pr["number"] == number:
                payload = deepcopy(pr)
                if not include_diff:
                    payload["changed_files"] = [
                        {k: v for k, v in f.items() if k != "patch"}
                        for f in payload["changed_files"]
                    ]
                return payload
        raise ToolExecutionError(
            f"No pull request #{number}.",
            details={"open_pull_requests": [pr["number"] for pr in self._data["pull_requests"]]},
        )

    def merged_since_last_release(self, *, repo: str | None = None) -> list[dict[str, Any]]:
        items = deepcopy(self._data["merged_since_last_release"])
        if repo:
            items = [pr for pr in items if pr["repo"] == repo or pr["repo"].endswith(f"/{repo}")]
        return items

    # -- continuous integration -------------------------------------------

    def ci_runs(self, *, repo: str | None = None, pull_request: int | None = None) -> list[dict]:
        items = deepcopy(self._data["ci_runs"])
        if repo:
            items = [r for r in items if r["repo"] == repo or r["repo"].endswith(f"/{repo}")]
        if pull_request is not None:
            items = [r for r in items if r.get("pull_request") == pull_request]
        return items

    def coverage(self, repo: str) -> dict[str, Any]:
        for report in self._data["coverage"]:
            if report["repo"] == repo or report["repo"].endswith(f"/{repo}"):
                return deepcopy(report)
        raise ToolExecutionError(
            f"No coverage report for {repo!r}.",
            details={"known_repos": [c["repo"] for c in self._data["coverage"]]},
        )

    def test_suite(self, repo: str) -> dict[str, Any]:
        for suite in self._data["test_suites"]:
            if suite["repo"] == repo or suite["repo"].endswith(f"/{repo}"):
                return deepcopy(suite)
        raise ToolExecutionError(
            f"No test suite registered for {repo!r}.",
            details={"known_repos": [s["repo"] for s in self._data["test_suites"]]},
        )

    # -- observability ------------------------------------------------------

    def deployments(self, *, service: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        items = deepcopy(self._data["deployments"])
        if service:
            items = [d for d in items if d["service"] == service]
        items.sort(key=lambda d: d["deployed_at"], reverse=True)
        return items[:limit]

    def incidents(self, *, status: str | None = None) -> list[dict[str, Any]]:
        items = deepcopy(self._data["incidents"])
        if status:
            items = [i for i in items if i["status"] == status]
        return items

    def metrics(self, service: str, *, metric: str | None = None) -> dict[str, Any]:
        series = self._data["metrics"].get(service)
        if series is None:
            raise ToolExecutionError(
                f"No metrics for service {service!r}.",
                details={"known_services": sorted(self._data["metrics"])},
            )
        if metric:
            if metric not in series:
                raise ToolExecutionError(
                    f"No metric {metric!r} for {service!r}.",
                    details={"available_metrics": sorted(series)},
                )
            return {"service": service, metric: deepcopy(series[metric])}
        return {"service": service, **deepcopy(series)}

    def logs(
        self,
        *,
        service: str | None = None,
        level: str | None = None,
        contains: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        items = deepcopy(self._data["logs"])
        if service:
            items = [entry for entry in items if entry["service"] == service]
        if level:
            items = [entry for entry in items if entry["level"] == level.upper()]
        if contains:
            needle = contains.lower()
            items = [entry for entry in items if needle in entry["message"].lower()]
        items.sort(key=lambda entry: entry["at"], reverse=True)
        return items[:limit]

    def oncall(self, *, team: str | None = None) -> list[dict[str, Any]]:
        items = deepcopy(self._data["oncall"])
        if team:
            items = [rota for rota in items if rota["team"] == team]
        return items

    # -- issue tracker ------------------------------------------------------

    def tickets(self, *, team: str | None = None, status: str | None = None) -> list[dict]:
        items = deepcopy(self._data["tickets"])
        if team:
            items = [t for t in items if t["team"] == team]
        if status:
            items = [t for t in items if t["status"].lower() == status.lower()]
        return items

    def ticket(self, ticket_id: str) -> dict[str, Any]:
        for ticket in self._data["tickets"]:
            if ticket["id"].lower() == ticket_id.lower():
                return deepcopy(ticket)
        raise ToolExecutionError(
            f"No ticket {ticket_id!r}.",
            details={"known_tickets": [t["id"] for t in self._data["tickets"]]},
        )

    def release_checklist(self) -> dict[str, Any]:
        return deepcopy(self._data["release_checklist"])


@lru_cache(maxsize=1)
def get_store() -> SystemsStore:
    with _SEED_PATH.open(encoding="utf-8") as handle:
        return SystemsStore(json.load(handle))
