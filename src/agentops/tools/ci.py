"""Continuous-integration and test-quality tools."""

from __future__ import annotations

from agentops.systems.store import SystemsStore
from agentops.tools.base import Tool, integer, obj, string


def ci_tools(store: SystemsStore) -> list[Tool]:
    """Tools over CI results, coverage and the repository's test conventions."""
    return [
        Tool(
            name="get_ci_runs",
            description=(
                "Fetch CI runs with per-job status and, for failures, the failing tests "
                "and their error messages."
            ),
            parameters=obj(
                {
                    "repo": string("Repository to filter by."),
                    "pull_request": integer("Only runs for this pull request number."),
                }
            ),
            # Models sometimes send numbers as strings ("482"); coerce so the filter still matches.
            handler=lambda repo=None, pull_request=None: store.ci_runs(
                repo=repo,
                pull_request=int(pull_request) if pull_request is not None else None,
            ),
            tags=["ci", "review", "testing", "release", "incident"],
        ),
        Tool(
            name="get_coverage_report",
            description=(
                "Line coverage for a repository's latest commit, against the baseline on main, "
                "broken down per file with the uncovered line numbers."
            ),
            parameters=obj({"repo": string("Repository name.")}, required=["repo"]),
            handler=lambda repo: store.coverage(repo),
            tags=["ci", "testing", "review"],
        ),
        Tool(
            name="get_test_suite",
            description=(
                "The test layout of a repository: test files, the conventions the team follows "
                "(frameworks, fixtures, how integration tests get their dependencies) and any "
                "known flaky tests. Read this before proposing new tests so they fit in."
            ),
            parameters=obj({"repo": string("Repository name.")}, required=["repo"]),
            handler=lambda repo: store.test_suite(repo),
            tags=["ci", "testing"],
        ),
    ]
