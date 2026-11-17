"""Runtime services: the run store, the trace bus replay logic and JSON logging."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from agentops.core.exceptions import RunNotFoundError
from agentops.core.logging import JsonFormatter
from agentops.orchestrator.models import Plan, RunResult, RunStatus
from agentops.runtime.events import EventBus, EventType
from agentops.runtime.store import InMemoryRunStore


def _run(run_id: str) -> RunResult:
    """A minimal finished run with the given id."""
    return RunResult(
        run_id=run_id, request="r", status=RunStatus.SUCCEEDED, answer="a", plan=Plan()
    )


async def test_the_store_keeps_recent_runs_and_evicts_the_oldest():
    """With room for two runs, saving a third drops the oldest and lists newest first."""
    store = InMemoryRunStore(max_size=2)
    for name in ("run_a", "run_b", "run_c"):
        await store.save(_run(name))

    assert await store.list_ids() == ["run_c", "run_b"]
    assert (await store.get("run_c")).run_id == "run_c"
    with pytest.raises(RunNotFoundError):
        await store.get("run_a")


async def test_a_subscriber_sees_everything_including_what_happened_before_it_joined():
    """A late subscriber gets the earlier events replayed, then the live ones, in order."""
    bus = EventBus("run_x")
    bus.emit(EventType.RUN_STARTED, message="before")

    received: list[EventType] = []

    async def consume() -> None:
        """Collect event types from the bus until it closes."""
        async for event in bus.subscribe():
            received.append(event.type)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    bus.emit(EventType.STEP_STARTED, message="during")
    bus.emit(EventType.RUN_COMPLETED, message="after")
    bus.close()
    await asyncio.wait_for(consumer, timeout=1)

    assert received == [
        EventType.RUN_STARTED,
        EventType.STEP_STARTED,
        EventType.RUN_COMPLETED,
    ]


async def test_no_event_is_delivered_twice_when_one_lands_mid_replay():
    """The replay window is the one place duplicates could appear."""
    bus = EventBus("run_x")
    for index in range(5):
        bus.emit(EventType.STEP_STARTED, message=str(index))

    stream = bus.subscribe()
    first = await anext(stream)
    bus.emit(EventType.STEP_COMPLETED, message="raced in")

    seen = [first.seq]
    bus.close()
    async for event in stream:
        seen.append(event.seq)

    assert seen == sorted(set(seen)), "sequence numbers must be unique and ordered"
    assert seen == [1, 2, 3, 4, 5, 6]


async def test_a_subscriber_joining_after_the_run_finished_still_gets_the_trace():
    """Subscribing to an already-closed bus replays the whole trace and then ends."""
    bus = EventBus("run_x")
    bus.emit(EventType.RUN_STARTED)
    bus.emit(EventType.RUN_COMPLETED)
    bus.close()

    seen = [event.type async for event in bus.subscribe()]
    assert seen == [EventType.RUN_STARTED, EventType.RUN_COMPLETED]


def test_the_json_formatter_keeps_extra_fields_and_never_collides():
    """Regression: `extra={"message": ...}` raises inside logging itself."""
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.run_id = "run_x"
    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello"
    assert payload["run_id"] == "run_x"
    assert payload["level"] == "INFO"
    assert "args" not in payload
