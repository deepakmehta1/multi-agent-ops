"""The endpoints that actually run the multi-agent system."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Path, Query
from fastapi.responses import StreamingResponse

from agentops.api.deps import OrchestratorDep, RunStoreDep
from agentops.core.logging import get_logger
from agentops.orchestrator.orchestrator import Orchestrator
from agentops.runtime.events import EventBus
from agentops.schemas import ErrorResponse, RunRequest, RunResponse

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["runs"])

_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "No OpenAI credentials available"},
    502: {"model": ErrorResponse, "description": "The model provider failed"},
}


@router.post(
    "/requests",
    response_model=RunResponse,
    responses=_ERRORS,
    summary="Handle a request with the multi-agent system",
)
async def create_run(
    payload: RunRequest,
    orchestrator: OrchestratorDep,
    runs: RunStoreDep,
) -> RunResponse:
    """Plan the work, delegate it, and return one synthesized answer.

    Blocks until the whole run finishes. Use `/v1/requests/stream` to watch it
    happen, which matters once several specialists are involved.
    """
    result = await orchestrator.run(payload.request)
    await runs.save(result)
    log.info(
        "run.finished",
        extra={
            "run_id": result.run_id,
            "status": str(result.status),
            "agents": result.agents_used,
            "duration_ms": result.duration_ms,
            "total_tokens": result.usage.total_tokens,
        },
    )
    return RunResponse.from_run(result, include_trace=payload.include_trace)


@router.post(
    "/requests/stream",
    responses={200: {"content": {"text/event-stream": {}}}, **_ERRORS},
    summary="Same, streamed as server-sent events",
)
async def stream_run(
    payload: RunRequest,
    orchestrator: OrchestratorDep,
    runs: RunStoreDep,
) -> StreamingResponse:
    """Stream the trace while the run is in flight.

    Emits one `event: trace` per decision -- plan created, step started, tool
    called -- and finally `event: result` with the same body `/v1/requests`
    returns. The client sees which agent is working before the answer exists.
    """
    # The id and the bus are created HERE, before the run starts, so the first SSE event can carry
    # the id and we can subscribe before anything is emitted.
    run_id = Orchestrator.new_run_id()
    bus = EventBus(run_id)

    async def publish() -> AsyncIterator[str]:
        """Yield the SSE stream: the run id, every trace event as it happens, then the result.

        Event order is always `run`, then zero or more `trace`, then `result`. The generator owns
        the lifetime of the orchestration task, which is what makes a client hang-up stop the work.
        """
        # Run the orchestration as its own task, so this generator can stream while it works.
        task = asyncio.create_task(orchestrator.run(payload.request, run_id=run_id, bus=bus))
        try:
            yield _sse("run", {"run_id": run_id, "request": payload.request})
            # Replays anything already emitted, then streams until the orchestrator closes the bus.
            async for event in bus.subscribe():
                yield _sse("trace", event.model_dump(mode="json"))
            # The bus is closed only when the run is over, so this returns promptly.
            result = await task
            await runs.save(result)
            yield _sse(
                "result",
                RunResponse.from_run(result, include_trace=payload.include_trace).model_dump(
                    mode="json"
                ),
            )
        except asyncio.CancelledError:
            # The client hung up; stop the run rather than letting it burn tokens.
            task.cancel()
            raise
        # Reached on normal completion AND when the client disconnects (the generator is closed).
        # Either way make sure no orchestration keeps spending tokens for nobody.
        finally:
            bus.close()
            if not task.done():
                task.cancel()

    return StreamingResponse(
        publish(),
        media_type="text/event-stream",
        # No caching, and tell nginx-style proxies not to buffer, or events arrive in one lump.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/runs",
    response_model=list[str],
    summary="Ids of recent runs, newest first",
)
async def list_runs(
    runs: RunStoreDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[str]:
    """Ids of the most recent runs, newest first (at most `limit`)."""
    return await runs.list_ids(limit=limit)


@router.get(
    "/runs/{run_id}",
    response_model=RunResponse,
    responses={404: {"model": ErrorResponse, "description": "No such run"}},
    summary="Fetch a finished run, including its full trace",
)
async def get_run(
    runs: RunStoreDep,
    run_id: str = Path(description="Run id returned by /v1/requests."),
    include_trace: bool = Query(default=True),
) -> RunResponse:
    """A finished run by id. Includes the full trace unless `include_trace=false`.

    Raises:
        RunNotFoundError: Unknown id, or the run aged out of the bounded history (404).
    """
    result = await runs.get(run_id)
    return RunResponse.from_run(result, include_trace=include_trace)


def _sse(event: str, data: dict) -> str:
    """Format one server-sent event.

    `data` is JSON on a single line (json.dumps never emits raw newlines), so one `data:`
    field per event is always enough. Non-JSON-native values are stringified.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
