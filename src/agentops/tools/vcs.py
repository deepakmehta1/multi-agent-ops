"""Version-control tools (a stand-in for the GitHub API)."""

from __future__ import annotations

from agentops.systems.store import SystemsStore
from agentops.tools.base import Tool, boolean, integer, obj, string


def vcs_tools(store: SystemsStore) -> list[Tool]:
    """Tools over source control: listing pull requests, reading one, and release scope."""
    return [
        Tool(
            name="list_pull_requests",
            description=(
                "List pull requests across the organisation's repositories. "
                "Returns metadata only -- use get_pull_request for the diff."
            ),
            parameters=obj(
                {
                    "repo": string("Repository, e.g. 'northwind/checkout-api' or 'checkout-api'."),
                    "state": string("Filter by state.", enum=["open", "merged", "closed"]),
                }
            ),
            handler=lambda repo=None, state=None: store.pull_requests(repo=repo, state=state),
            tags=["vcs", "review", "release"],
        ),
        Tool(
            name="get_pull_request",
            description=(
                "Fetch one pull request: description, reviewers, comments and changed files. "
                "Set include_diff=true to also get the patch for each file -- do that when you "
                "need to judge the code itself."
            ),
            parameters=obj(
                {
                    "number": integer("Pull request number, e.g. 482."),
                    "include_diff": boolean("Include the patch for each changed file."),
                },
                required=["number"],
            ),
            # Diffs are large and cost tokens, so they are opt-in: a model only asks for one when it
            # really has to judge the code, not merely to list or triage pull requests.
            handler=lambda number, include_diff=False: store.pull_request(
                int(number), include_diff=bool(include_diff)
            ),
            tags=["vcs", "review", "testing"],
        ),
        Tool(
            name="list_merged_pull_requests",
            description=(
                "List pull requests merged since the last production release, with their "
                "labels and linked tickets. The input for release notes."
            ),
            parameters=obj({"repo": string("Restrict to one repository.")}),
            handler=lambda repo=None: store.merged_since_last_release(repo=repo),
            tags=["vcs", "release"],
        ),
    ]
