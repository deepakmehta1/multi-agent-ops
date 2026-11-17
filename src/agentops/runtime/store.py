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
    async def save(self, run: RunResult) -> None: ...

    async def get(self, run_id: str) -> RunResult: ...

    async def list_ids(self, limit: int = 50) -> list[str]: ...


class InMemoryRunStore:
    def __init__(self, max_size: int = 200) -> None:
        self._runs: OrderedDict[str, RunResult] = OrderedDict()
        self._max_size = max_size

    async def save(self, run: RunResult) -> None:
        self._runs[run.run_id] = run
        self._runs.move_to_end(run.run_id)
        while len(self._runs) > self._max_size:
            self._runs.popitem(last=False)

    async def get(self, run_id: str) -> RunResult:
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunNotFoundError(f"No run with id {run_id!r}.") from None

    async def list_ids(self, limit: int = 50) -> list[str]:
        return list(reversed(list(self._runs)))[:limit]
