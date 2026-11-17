"""Builds the one registry every agent draws its tools from."""

from __future__ import annotations

from agentops.systems.store import SystemsStore, get_store
from agentops.tools.base import ToolRegistry
from agentops.tools.ci import ci_tools
from agentops.tools.observability import observability_tools
from agentops.tools.tracker import tracker_tools
from agentops.tools.vcs import vcs_tools


def build_registry(store: SystemsStore | None = None) -> ToolRegistry:
    """Assemble the registry of every tool, all backed by one `SystemsStore`.

    Args:
        store: The backing systems. Defaults to the cached seeded store; tests pass their own
            so they never depend on (or mutate) shared state.
    """
    store = store or get_store()
    registry = ToolRegistry()
    for tool in (
        *vcs_tools(store),
        *ci_tools(store),
        *observability_tools(store),
        *tracker_tools(store),
    ):
        registry.register(tool)
    return registry
