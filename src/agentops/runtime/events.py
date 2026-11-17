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
    """The kinds of event on a run's trace, and what each means.

    Name              Emitted when                          `data` carries
    ----------------  ------------------------------------  ---------------------------
    run.started       the run begins (message = request)    -
    plan.created      the planner returned a plan           direct, steps
    plan.revised      follow-up work was scheduled          steps
    step.started      a specialist begins a delegation      depends_on
    agent.thinking    a specialist is about to call tools   tools
    tool.called       one tool call begins                  arguments
    tool.result       that call returned                    ok, duration_ms, preview
    step.completed    a specialist delivered its report     tool_calls, tokens, truncated
    step.failed       a step errored or timed out           -  (message = reason)
    step.skipped      a step was not run (upstream failed)  -  (message = reason)
    synthesis.started the answer is being written           -
    run.completed     the run finished (any outcome)        steps, delegated
    run.failed        the run could not complete            -  (message = reason)
    """

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
    """One entry on a run's trace; the same object feeds the SSE stream, the response and logs.

    `seq` orders events within a run, `agent` and `step_id` say who and which delegation,
    `message` is a human-readable line, and `data` holds the event-specific fields.
    """

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
        """Create an empty bus for one run."""
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
        """Record an event and push it to every live subscriber.

        Synchronous and non-blocking: events go into unbounded queues with `put_nowait`, so a
        slow or stalled client can never slow the orchestrator down. The event is also written
        to the structured log, which is why the trace and the logs always agree.

        Args:
            event_type: What happened.
            agent: Who it happened to (an agent name, or "orchestrator").
            step_id: The plan step involved, if any.
            message: A short human-readable line.
            **data: Event-specific fields, stored on `TraceEvent.data`.
        """
        # Per-run, monotonic. `subscribe` relies on it to drop duplicates during replay.
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
        # History is kept even with no subscribers, so a non-streaming call still returns the full
        # trace.
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
        # Register BEFORE snapshotting history: an event emitted in between then lands in both the
        # replay and the queue (a duplicate, filtered by `seq` below) instead of being lost.
        self._subscribers.add(queue)
        # Captured now, not after the replay. A bus that was ALREADY closed never delivers a
        # sentinel to this queue, so the replay is all there is. A bus that closes MID-replay
        # does: its sentinel and trailing events are in our queue and must be drained, not
        # abandoned.
        # Captured now, not after the replay: a bus that closes MID-replay still has its sentinel
        # and trailing events in our queue and must be drained, not abandoned.
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
                # Already delivered by the replay above.
                if queued.seq <= replayed:
                    continue
                yield queued
        finally:
            self._subscribers.discard(queue)
