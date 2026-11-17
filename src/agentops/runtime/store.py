"""Where finished runs live.

`InMemoryRunStore` is a bounded dict -- enough to serve GET /v1/runs/{id} in a
single process. Swap in a Redis or Postgres implementation of `RunStore` and
nothing above this module changes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol

from agentops.core.exceptions import RunNotFoundError

if TYPE_CHECKING:
    from agentops.orchestrator.models import RunResult


class RunStore(Protocol):
    """Where finished runs are kept. Implement these three methods to plug in another backend."""

    async def save(self, run: RunResult) -> None:
        """Store a finished run, replacing any run with the same id."""
        ...

    async def get(self, run_id: str) -> RunResult:
        """Return a stored run.

        Raises:
            RunNotFoundError: No such run, or it has aged out.
        """
        ...

    async def list_ids(self, limit: int = 50) -> list[str]:
        """Run ids, newest first, at most `limit`."""
        ...


class InMemoryRunStore:
    """A bounded in-process store. Runs are lost on restart and not shared between workers."""

    def __init__(self, max_size: int = 200) -> None:
        """Keep at most `max_size` runs; the oldest are dropped first."""
        self._runs: OrderedDict[str, RunResult] = OrderedDict()
        self._max_size = max_size

    async def save(self, run: RunResult) -> None:
        """Store a run, evicting the oldest ones if that exceeds `max_size`.

        Re-saving an id moves it to the newest position.
        """
        self._runs[run.run_id] = run
        self._runs.move_to_end(run.run_id)
        while len(self._runs) > self._max_size:
            self._runs.popitem(last=False)

    async def get(self, run_id: str) -> RunResult:
        """Return a stored run.

        Raises:
            RunNotFoundError: No run with that id (never stored, or evicted).
        """
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunNotFoundError(f"No run with id {run_id!r}.") from None

    async def list_ids(self, limit: int = 50) -> list[str]:
        """Run ids, newest first, at most `limit`."""
        return list(reversed(list(self._runs)))[:limit]
