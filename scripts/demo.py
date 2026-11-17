#!/usr/bin/env python
"""Watch the orchestrator work.

Two modes:

    python scripts/demo.py --offline      # no API key, no server: a scripted run
    python scripts/demo.py "Review PR 482" # streams from a running server

The offline mode exists so the system can be demonstrated, and its control flow
followed, without spending a token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Lets `python scripts/demo.py --offline` work from a fresh checkout, before the package is pip-
# installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_REQUEST = (
    "PR 482 on checkout-api is up for review and we want it in Thursday's release. "
    "Tell me whether it is safe to merge, what tests it still needs, and whether the "
    "release can actually go out."
)

ICONS = {
    "run.started": "▶",
    "plan.created": "🗂",
    "plan.revised": "↻",
    "step.started": "→",
    "step.completed": "✓",
    "step.failed": "✗",
    "step.skipped": "⊘",
    "agent.thinking": "…",
    "tool.called": "  ⚙",
    "tool.result": "  ↩",
    "synthesis.started": "∑",
    "run.completed": "■",
    "run.failed": "✗",
}


def render(event: dict) -> str:
    """Format one trace event as a single terminal line: icon, agent, step id, message.

    Long messages are clipped so one event never wraps the screen.
    """
    icon = ICONS.get(event["type"], "·")
    who = event.get("agent") or "-"
    step = f"[{event['step_id']}] " if event.get("step_id") else ""
    message = (event.get("message") or "").strip().replace("\n", " ")
    if len(message) > 110:
        message = message[:107] + "..."
    return f"{icon} {who:<20} {step}{message}"


# --- offline ---------------------------------------------------------------


async def run_offline(request: str) -> int:
    """Run a complete orchestration in-process against a scripted model: no key, no network.

    Everything except the model is real - the planner, the dependency-graph executor, the
    agent tool loops and the tools reading the seeded data - so this shows the genuine
    control flow, including s1 and s2 running in parallel with s3 waiting for both.
    Only the words the "model" says are canned.

    Args:
        request: The request to run (the scripted plan ignores its wording).

    Returns:
        Process exit code (0).
    """
    from agentops.agents.registry import AgentRegistry
    from agentops.core.config import Settings
    from agentops.core.logging import configure_logging
    from agentops.llm.fake import Reply, ScriptedLLM
    from agentops.orchestrator.orchestrator import Orchestrator
    from agentops.tools.catalog import build_registry

    configure_logging("WARNING", "text")

    plan = {
        "reasoning": "Three independent questions, one per specialist.",
        "answer_directly": None,
        "steps": [
            {
                "id": "s1",
                "agent": "code_reviewer",
                "objective": "Review PR 482 on checkout-api for correctness.",
                "depends_on": [],
            },
            {
                "id": "s2",
                "agent": "test_engineer",
                "objective": "Assess test coverage for PR 482 and name the missing tests.",
                "depends_on": [],
            },
            {
                "id": "s3",
                "agent": "release_manager",
                "objective": "Decide whether 2.14.0 can ship on Thursday.",
                # s3 waits for BOTH s1 and s2; s1 and s2 have no dependencies, so they run in
                # parallel.
                "depends_on": ["s1", "s2"],
            },
        ],
    }

    def is_planner(messages, tools):
        """True for the planner's call: JSON mode, no tools offered."""
        return not tools and "Reply with JSON only" in (messages[0].content or "")

    def is_replan(messages, tools):
        """True for the planner's follow-up call after results are in."""
        return is_planner(messages, tools) and any(
            "Is anything still missing" in (m.content or "") for m in messages
        )

    def agent_is(title):
        """Match a call made by the specialist with this job title."""
        return lambda messages, _tools: f"You are the {title}" in (messages[0].content or "")

    def used_tools(messages, _tools):
        """True once the conversation contains at least one tool result."""
        return any(m.role == "tool" for m in messages)

    # RULE ORDER MATTERS. A rule fires once, in registration order. Each specialist's rule returns
    # its tool calls the first time it is asked; after that it is spent, so the sticky `used_tools`
    # rule below turns the following turn (which now holds tool results) into a report.
    llm = ScriptedLLM()
    llm.when(
        lambda m, t: is_planner(m, t) and not is_replan(m, t),
        Reply(content=json.dumps(plan)),
    )
    llm.when(
        is_replan,
        Reply(content=json.dumps({"reasoning": "complete", "steps": []})),
        sticky=True,
    )
    llm.when(
        agent_is("Staff Engineer doing code review"),
        Reply(
            tool_calls=[
                ("get_pull_request", {"number": 482, "include_diff": True}),
                ("get_ci_runs", {"pull_request": 482}),
            ]
        ),
    )
    llm.when(
        agent_is("Software Engineer in Test"),
        Reply(
            tool_calls=[
                ("get_test_suite", {"repo": "checkout-api"}),
                ("get_coverage_report", {"repo": "checkout-api"}),
            ]
        ),
    )
    llm.when(
        agent_is("Release Manager"),
        Reply(tool_calls=[("get_release_checklist", {}), ("list_incidents", {"status": "open"})]),
    )
    llm.when(used_tools, Reply(content="(scripted specialist report)"), sticky=True)
    llm.when(
        lambda messages, tools: (
            not tools and "you answer the person who asked" in (messages[0].content or "")
        ),
        Reply(content="(scripted final answer -- supply an API key to get a real one)"),
        sticky=True,
    )

    orchestrator = Orchestrator(
        llm=llm,
        agents=AgentRegistry(),
        tools=build_registry(),
        settings=Settings(openai_api_key=None, log_format="text"),
    )
    result = await orchestrator.run(request)

    print(f"\n\033[1mRequest\033[0m\n{request}\n\n\033[1mTrace\033[0m")
    for event in result.trace:
        print(render(event.model_dump(mode="json")))
    print(f"\n\033[1mAnswer\033[0m\n{result.answer}")
    print(
        f"\n\033[2mstatus={result.status} agents={', '.join(result.agents_used)} "
        f"steps={len(result.steps)} tokens={result.usage.total_tokens}\033[0m"
    )
    return 0


# --- live ------------------------------------------------------------------


async def run_live(request: str, base_url: str, api_key: str | None) -> int:
    """Stream a run from a live server and print each trace event as it arrives.

    Args:
        request: The request to send.
        base_url: Where the API is listening.
        api_key: Sent as `X-OpenAI-Key` when given; otherwise the server's key is used.

    Returns:
        0 on success, 1 if the server refused the request or could not be reached.
    """
    import httpx

    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["X-OpenAI-Key"] = api_key

    print(f"\n\033[1mRequest\033[0m\n{request}\n\n\033[1mTrace\033[0m")
    async with httpx.AsyncClient(base_url=base_url, timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                "/v1/requests/stream",
                json={"request": request},
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode()
                    print(f"\n{response.status_code}: {body}", file=sys.stderr)
                    return 1
                name = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        name = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        payload = json.loads(line.removeprefix("data: "))
                        if name == "trace":
                            print(render(payload))
                        elif name == "result":
                            print(f"\n\033[1mAnswer\033[0m\n{payload['answer']}")
                            print(
                                f"\n\033[2mstatus={payload['status']} "
                                f"agents={', '.join(payload['agents_used'])} "
                                f"tokens={payload['usage']['total_tokens']} "
                                f"took={payload['duration_ms']}ms\033[0m"
                            )
        except httpx.ConnectError:
            print(f"\nCannot reach {base_url}. Start the server with `make run`.", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    """Parse arguments and run offline or live; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", default=DEFAULT_REQUEST)
    parser.add_argument(
        "--offline", action="store_true", help="Run in-process with a scripted LLM."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None, help="Sent as X-OpenAI-Key.")
    args = parser.parse_args()

    if args.offline:
        return asyncio.run(run_offline(args.request))
    return asyncio.run(run_live(args.request, args.base_url, args.api_key))


if __name__ == "__main__":
    raise SystemExit(main())
