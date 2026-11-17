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
    """Read-only view over the seeded Northwind Payments data.

    Two rules apply to every method:

    * Everything returned is a deep copy, so a caller (or a model-driven tool) can never
      mutate the shared seed and corrupt the next run.
    * A lookup that misses raises `ToolExecutionError` whose `details` list the valid
      values, so the model sees what it should have asked for and can retry.

    Repository filters accept either the full name (`northwind/checkout-api`) or the short
    one (`checkout-api`), because models and people use both interchangeably.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Wrap an already-loaded seed document (see `get_store`)."""
        self._data = data

    # -- organisation ------------------------------------------------------

    def organisation(self) -> dict[str, Any]:
        """Company facts: delivery teams and the release train schedule."""
        return deepcopy(self._data["organisation"])

    def services(self) -> list[dict[str, Any]]:
        """Every service with its owning team and repository."""
        return deepcopy(self._data["services"])

    def service(self, name: str) -> dict[str, Any]:
        """One service by exact name.

        Raises:
            ToolExecutionError: No such service; `details.known_services` lists the real ones.
        """
        for service in self._data["services"]:
            if service["name"] == name:
                return deepcopy(service)
        raise ToolExecutionError(
            f"Unknown service {name!r}.",
            details={"known_services": [s["name"] for s in self._data["services"]]},
        )

    # -- version control ---------------------------------------------------

    def pull_requests(self, *, repo: str | None = None, state: str | None = None) -> list[dict]:
        """List pull requests as metadata only.

        The changed-file list is dropped on purpose: a listing across the whole organisation
        would otherwise carry every file and patch. Use `pull_request` for the detail.

        Args:
            repo: Keep only this repository (short or full name).
            state: Keep only "open", "merged" or "closed" pull requests.
        """
        items = self._data["pull_requests"]
        if repo:
            items = [pr for pr in items if pr["repo"] == repo or pr["repo"].endswith(f"/{repo}")]
        if state:
            items = [pr for pr in items if pr["state"] == state]
        # Listing view: metadata only (no changed files).
        return [{k: v for k, v in deepcopy(pr).items() if k != "changed_files"} for pr in items]

    def pull_request(self, number: int, *, include_diff: bool = False) -> dict[str, Any]:
        """One pull request by number, with comments, reviewers and changed files.

        Args:
            number: The pull request number.
            include_diff: Keep each file's `patch`. Off by default because patches dominate the
                token cost and are only needed to judge the code itself.

        Raises:
            ToolExecutionError: No such pull request; `details` lists the known numbers.
        """
        for pr in self._data["pull_requests"]:
            if pr["number"] == number:
                payload = deepcopy(pr)
                if not include_diff:
                    # Strip each file's patch unless the caller asked for the diff.
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
        """Pull requests merged since the last production release, the input for release notes.

        Args:
            repo: Keep only this repository (short or full name).
        """
        items = deepcopy(self._data["merged_since_last_release"])
        if repo:
            items = [pr for pr in items if pr["repo"] == repo or pr["repo"].endswith(f"/{repo}")]
        return items

    # -- continuous integration -------------------------------------------

    def ci_runs(self, *, repo: str | None = None, pull_request: int | None = None) -> list[dict]:
        """CI runs, including the failing tests and their messages for failed jobs.

        Args:
            repo: Keep only this repository (short or full name).
            pull_request: Keep only runs for this pull request number.
        """
        items = deepcopy(self._data["ci_runs"])
        if repo:
            items = [r for r in items if r["repo"] == repo or r["repo"].endswith(f"/{repo}")]
        if pull_request is not None:
            items = [r for r in items if r.get("pull_request") == pull_request]
        return items

    def coverage(self, repo: str) -> dict[str, Any]:
        """Line coverage for a repository against the baseline on main, per file.

        Raises:
            ToolExecutionError: No report for that repository; `details` lists those that exist.
        """
        for report in self._data["coverage"]:
            if report["repo"] == repo or report["repo"].endswith(f"/{repo}"):
                return deepcopy(report)
        raise ToolExecutionError(
            f"No coverage report for {repo!r}.",
            details={"known_repos": [c["repo"] for c in self._data["coverage"]]},
        )

    def test_suite(self, repo: str) -> dict[str, Any]:
        """Test layout, team conventions and known flaky tests for one repository.

        Raises:
            ToolExecutionError: Nothing registered for that repository.
        """
        for suite in self._data["test_suites"]:
            if suite["repo"] == repo or suite["repo"].endswith(f"/{repo}"):
                return deepcopy(suite)
        raise ToolExecutionError(
            f"No test suite registered for {repo!r}.",
            details={"known_repos": [s["repo"] for s in self._data["test_suites"]]},
        )

    # -- observability ------------------------------------------------------

    def deployments(self, *, service: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Production deployments, newest first.

        Args:
            service: Keep only this service.
            limit: Maximum number returned, after sorting.
        """
        items = deepcopy(self._data["deployments"])
        if service:
            items = [d for d in items if d["service"] == service]
        # ISO-8601 UTC timestamps sort chronologically as plain strings, so no parsing is needed.
        items.sort(key=lambda d: d["deployed_at"], reverse=True)
        return items[:limit]

    def incidents(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Incidents with severity, summary and timeline.

        Args:
            status: Keep only "open" or "resolved" incidents.
        """
        items = deepcopy(self._data["incidents"])
        if status:
            items = [i for i in items if i["status"] == status]
        return items

    def metrics(self, service: str, *, metric: str | None = None) -> dict[str, Any]:
        """Time series for one service: either every series it reports, or just one.

        Args:
            service: The service name.
            metric: A single series, e.g. `p99_latency_ms`. Omit for all of them.

        Raises:
            ToolExecutionError: Unknown service or metric; `details` lists the valid choices.
        """
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
            # A single requested series.
            return {"service": service, metric: deepcopy(series[metric])}
        # No metric named: return every series the service reports.
        return {"service": service, **deepcopy(series)}

    def logs(
        self,
        *,
        service: str | None = None,
        level: str | None = None,
        contains: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Aggregated log lines, newest first.

        Each entry has a `count`, so one line can stand for thousands of occurrences.

        Args:
            service: Keep only this service.
            level: Keep only this level; matched case-insensitively.
            contains: Keep only messages containing this text, case-insensitively.
            limit: Maximum entries returned, after filtering and sorting.
        """
        items = deepcopy(self._data["logs"])
        if service:
            items = [entry for entry in items if entry["service"] == service]
        if level:
            items = [entry for entry in items if entry["level"] == level.upper()]
        if contains:
            needle = contains.lower()
            items = [entry for entry in items if needle in entry["message"].lower()]
        # Newest first: when debugging, the most recent lines matter most, and `limit` then keeps
        # them. Filtering happens before the cut so a match is never lost to an unrelated line.
        items.sort(key=lambda entry: entry["at"], reverse=True)
        return items[:limit]

    def oncall(self, *, team: str | None = None) -> list[dict[str, Any]]:
        """Who is on call, primary and secondary.

        Args:
            team: Keep only this team's rota.
        """
        items = deepcopy(self._data["oncall"])
        if team:
            items = [rota for rota in items if rota["team"] == team]
        return items

    # -- issue tracker ------------------------------------------------------

    def tickets(self, *, team: str | None = None, status: str | None = None) -> list[dict]:
        """Delivery tickets.

        Args:
            team: Keep only this team's tickets.
            status: Keep only this status, matched case-insensitively (e.g. "in review").
        """
        items = deepcopy(self._data["tickets"])
        if team:
            items = [t for t in items if t["team"] == team]
        if status:
            items = [t for t in items if t["status"].lower() == status.lower()]
        return items

    def ticket(self, ticket_id: str) -> dict[str, Any]:
        """One ticket by id, matched case-insensitively ("del-1123" finds "DEL-1123").

        Raises:
            ToolExecutionError: No such ticket; `details` lists the known ids.
        """
        for ticket in self._data["tickets"]:
            if ticket["id"].lower() == ticket_id.lower():
                return deepcopy(ticket)
        raise ToolExecutionError(
            f"No ticket {ticket_id!r}.",
            details={"known_tickets": [t["id"] for t in self._data["tickets"]]},
        )

    def release_checklist(self) -> dict[str, Any]:
        """The gate for the next release: target date and every item with its status and notes."""
        return deepcopy(self._data["release_checklist"])


@lru_cache(maxsize=1)
def get_store() -> SystemsStore:
    """Load the seed once and share one `SystemsStore` for the process.

    Cached, so every tool and every run reads the same data without re-parsing the file.
    Safe to share because the store only ever hands out copies.
    """
    with _SEED_PATH.open(encoding="utf-8") as handle:
        return SystemsStore(json.load(handle))
