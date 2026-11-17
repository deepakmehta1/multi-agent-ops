# Architecture

Why this system is shaped the way it is. [README.md](README.md) covers what it does and how to run
it; this document is about the decisions.

## The problem

A delivery team's questions are rarely answerable by one role. *"Can PR 482 go into Thursday's
release?"* needs someone to read the diff, someone to judge the tests, and someone who owns the
release gate — and the third person's answer depends on the first two. A single prompt with every
tool attached handles this badly: the context fills with irrelevant tool output, the model loses the
thread of which sub-question it was answering, and there is no way to run the independent parts at
once.

So the work is split the way the team is: one orchestrator that decides who does what, and
specialists who each see only their own job.

## The run lifecycle

```
                 ┌──────────────────────────────────────────────┐
  request ─────▶ │  Planner                                     │
                 │  roster mandates ──▶ validated plan (a DAG)  │
                 └──────────────────┬───────────────────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │  Executor              │   wave 1: s1 ∥ s2
                        │  topological waves,    │   wave 2: s3 (needs s1, s2)
                        │  bounded concurrency   │
                        └───────────┬────────────┘
                                    │  step results
                        ┌───────────▼────────────┐
                        │  Review (replan)       │──── follow-up steps ──┐
                        └───────────┬────────────┘                       │
                                    │ complete                           │
                        ┌───────────▼────────────┐                       │
                        │  Synthesizer           │◀──────────────────────┘
                        └───────────┬────────────┘
                                    ▼
                        answer + plan + reports + trace
```

### 1. Planning

`Planner` ([`orchestrator/planner.py`](src/agentops/orchestrator/planner.py)) makes one JSON-mode
call and gets back a plan: steps, each with an agent, an objective and `depends_on`.

Three things make this work in practice:

**The planner sees mandates, not capabilities.** Its prompt contains each agent's `mandate` — one or
two sentences including the kinds of request it covers — and the names of its tools. It does not see
the specialists' instructions or output contracts. Delegating is a routing decision, and giving the
router the full prompts of everyone it might route to both costs tokens and invites it to do their
job instead.

**Objectives must be self-contained.** A specialist cannot see the other steps' objectives, only the
*outputs* of the steps it depends on. The planner is told this explicitly, which is what stops
objectives like "review it" that are meaningless out of context.

**Dependencies are opt-in.** An empty `depends_on` is the documented default, and the prompt gives
the distinction with examples: reviewing PR 482 and assessing its coverage are independent, but
deciding go/no-go after finding what blocks the release is not. Without that pressure, models chain
everything serially and the parallelism is never used.

Then the plan is validated in code before anything runs: unknown agent, duplicate id, empty
objective, dangling dependency, self-dependency, dependency cycle (Kahn's algorithm), or more than
`MAX_PLAN_STEPS` steps. An invalid plan is sent back once with the specific error; a second failure
fails the run. One repair attempt, not a loop — a planner that cannot produce a valid graph twice
will not produce one on the fifth try, and the retry budget is the user's money.

The planner may also set `answer_directly` and return no steps, for questions like "when is the
release train?" that need no specialist. The orchestrator then returns immediately — one model call
for the whole request.

### 2. Execution

`Executor` ([`orchestrator/executor.py`](src/agentops/orchestrator/executor.py)) walks the graph in
topological waves. Each wave is every step whose dependencies are now satisfied, run with
`asyncio.gather` under a semaphore of `MAX_PARALLEL_STEPS`.

Failure handling is the part worth explaining. A step can fail three ways: the model provider errors,
the step exceeds `STEP_TIMEOUT_SECONDS`, or something unexpected raises. In all three the step is
recorded as `failed` with its reason, and **its dependents are marked `skipped` rather than run** —
running the release manager on a missing incident report would produce a confident answer built on
nothing. Steps in other branches of the graph are unaffected, the run completes as `partial`, and
the synthesizer is told which reports are missing so the final answer says what is unverified.

A step's dependencies are passed as context keyed by agent name, so a downstream specialist sees
*"### From incident_responder"* followed by that report, plus an instruction not to redo work that
is already done.

### 3. Review

After the first wave of results the orchestrator gets one chance (`MAX_REPLAN_ROUNDS`) to look at
what came back and schedule follow-up steps. It is given the original request and every report, and
asked whether anything is still missing — a failed step worth retrying differently, or a gap the
results exposed.

This is what makes the system agentic rather than a fancy router. "The review found an untested
concurrency path, so now write that test" is not a hard-coded pipeline stage; it emerges because the
orchestrator reads the review and decides to delegate again. Follow-up steps may depend on steps
from the earlier round, which is why `Executor.execute` takes `prior`.

It is capped at one round by default because the marginal value drops fast while the cost does not.

### 4. Synthesis

The synthesizer is given the request and every report, and writes the answer. Its prompt forbids
adding facts, requires it to surface disagreement between specialists rather than silently picking
one, requires it to name what is unverified when a step failed, and tells it to keep the
specialists' concrete artefacts — code blocks, release-note drafts — intact rather than summarising
them away.

**A single successful report skips synthesis entirely** and is returned verbatim. Asking a model to
rewrite one report it was just handed only loses detail and costs a call.

## The specialist loop

Every specialist is an instance of the same `Agent`
([`agents/base.py`](src/agentops/agents/base.py)). What differs is its prompt, its tool list and its
output contract — not its code path. Adding the fifth specialist is a file, not a refactor.

The loop: call the model with the agent's tools; if it returns tool calls, execute them (in parallel
— `asyncio.gather` over the calls in one turn) and feed the results back; repeat until the model
answers without calling a tool, or until `MAX_AGENT_ITERATIONS`. On exhaustion the agent gets one
final call *with the tools removed* and an instruction to report from what it has and state what it
could not verify. A truncated report is flagged as such all the way into the response.

Two deliberate choices:

**Tool access is enforced in code.** A tool call is checked against the agent's own `tool_names`,
not just the registry. If the release manager's model asks for `search_logs`, the call is refused
with the list of tools it does actually have, and that refusal goes back into its context. Prompts
are not a security boundary; this is how mandates stay real.

**A tool error is information, not a crash.** `Tool.invoke` turns every failure — unknown argument,
missing required argument, a raised `ToolExecutionError`, even an unexpected exception — into a JSON
result the model reads. Errors carry `details` that name the valid inputs, so
`get_service_metrics("nope")` returns the list of real service names and the model's next call is
usually right. An exception thrown through the loop would waste the whole step instead.

## The trace

Every decision is published to an `EventBus`
([`runtime/events.py`](src/agentops/runtime/events.py)): run started, plan created, plan revised,
step started/completed/failed/skipped, agent thinking, tool called, tool result, synthesis started,
run completed/failed. One stream feeds three consumers — the SSE endpoint while the run is in
flight, the `trace` field of the final response, and the structured logs — and nothing in the
orchestrator knows which of them is listening.

Events carry a monotonic `seq`. A subscriber is registered **before** history is snapshotted, so an
event emitted during that window is captured twice rather than lost, and the sequence number filters
the duplicate out. A subscriber that joins after the run finished gets the full history and
terminates; one that joins mid-replay still receives the events that arrived while it was replaying.
That last case is a one-line bug that silently truncates a client's view of a run, so it has a test
of its own.

On the HTTP side, a client disconnecting cancels the run rather than leaving it to finish into a
socket nobody is reading.

## Layer boundaries

```
api/            HTTP: routing, dependency wiring, error → status mapping
  ↓
orchestrator/   planning, DAG execution, review, synthesis
  ↓
agents/         the specialist loop; one declarative file per role
  ↓
tools/          named, schema'd, per-agent-scoped capabilities
  ↓
systems/        the integrations — today, seeded fixtures
  ↓
llm/            the one seam to any model provider
```

Each boundary is there to make one specific change cheap:

- **`llm/`** — `LLMClient` is a one-method protocol. Swapping providers, or substituting the
  scripted fake that every test runs against, touches nothing above it.
- **`systems/`** — every tool goes through `SystemsStore`. Replacing fixtures with the real GitHub,
  Datadog and Jira APIs is one file; the tools, agents and prompts are already written as they would
  be in production.
- **`runtime/store.py`** — `RunStore` is a protocol with a bounded in-memory implementation. Redis
  or Postgres is a new class, not a change to the routes.
- **`core/exceptions.py`** — domain exceptions carry their own status code and error code, mapped
  once in `api/errors.py`, so no route handler writes HTTP semantics.

## Cost and failure, concretely

An agentic system's two unbounded failure modes are spend and hang, and every loop here has a
ceiling: steps per plan, replan rounds, tool-calling turns per agent, concurrent steps, seconds per
step, seconds per run. Exceeding any of them is reported in the response and the trace — a truncated
agent is flagged, a timed-out step is `failed` with its reason, an over-budget run returns `failed`
with the budget it blew. Nothing is silently swallowed, because a quietly degraded answer is worse
than an honest partial one.

Model selection follows the same logic: the orchestrator does the reasoning-heavy work and gets the
better model; the specialists make most of the calls and get the cheaper one. Both are per-deployment
environment variables.

Retries sit in `OpenAIClient`, not in the agent loop, with jittered exponential backoff for
connection errors, rate limits and 5xx — and no retry at all for a 400 or a revoked key, which will
fail identically the second time.
