"""The trace: every decision the system makes is published as an event.

The same stream feeds three consumers -- the SSE endpoint while a run is in
flight, the `trace` field of the final response, and the structured logs.
Nothing in the orchestrator has to know which of those is listening.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agentops.core.logging import get_logger

log = get_logger("agentops.trace")


class EventType(StrEnum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    PLAN_CREATED = "plan.created"
    PLAN_REVISED = "plan.revised"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"
    AGENT_THINKING = "agent.thinking"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    SYNTHESIS_STARTED = "synthesis.started"


class TraceEvent(BaseModel):
    seq: int = 0
    """Monotonic position in the run's trace; used to de-duplicate a replay."""
    type: EventType
    run_id: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent: str | None = None
    step_id: str | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Fan-out of trace events to zero or more live subscribers.

    Events are always appended to `history`, so a run started before anyone
    subscribed still returns its full trace.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.history: list[TraceEvent] = []
        self._subscribers: set[asyncio.Queue[TraceEvent | None]] = set()
        self._closed = False
        self._seq = 0

    def emit(
        self,
        event_type: EventType,
        *,
        agent: str | None = None,
        step_id: str | None = None,
        message: str | None = None,
        **data: Any,
    ) -> TraceEvent:
        self._seq += 1
        event = TraceEvent(
            seq=self._seq,
            type=event_type,
            run_id=self.run_id,
            agent=agent,
            step_id=step_id,
            message=message,
            data=data,
        )
        self.history.append(event)
        log.info(
            str(event_type),
            extra={
                "run_id": self.run_id,
                "agent": agent,
                "step_id": step_id,
                "event": str(event_type),
            },
        )
        for queue in list(self._subscribers):
            queue.put_nowait(event)
        return event

    def close(self) -> None:
        """Signal subscribers that no further events will arrive."""
        self._closed = True
        for queue in list(self._subscribers):
            queue.put_nowait(None)

    async def subscribe(self) -> AsyncIterator[TraceEvent]:
        """Replay what has happened so far, then stream the rest.

        The queue is registered *before* history is snapshotted, so an event
        emitted during that window is captured twice rather than lost; the
        sequence number then filters the duplicate out.
        """
        queue: asyncio.Queue[TraceEvent | None] = asyncio.Queue()
        self._subscribers.add(queue)
        # A bus that was already closed will never deliver a sentinel to this
        # queue, so the replay is all there is.
        already_closed = self._closed
        try:
            replayed = 0
            for event in list(self.history):
                replayed = event.seq
                yield event
            if already_closed:
                return
            while True:
                queued = await queue.get()
                if queued is None:
                    return
                if queued.seq <= replayed:
                    continue
                yield queued
        finally:
            self._subscribers.discard(queue)
