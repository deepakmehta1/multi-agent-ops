"""The planning half of the orchestrator.

Given a request and the roster, it produces a dependency graph of delegations --
or decides no specialist is needed. The plan is validated before anything runs,
and the model gets exactly one chance to repair an invalid one.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from agentops.agents.registry import AgentRegistry
from agentops.core.config import Settings
from agentops.core.exceptions import PlanValidationError
from agentops.core.logging import get_logger
from agentops.llm.base import LLMClient, Message, Usage
from agentops.orchestrator.models import Plan, StepResult
from agentops.runtime.events import EventBus, EventType

log = get_logger(__name__)

_PLAN_SCHEMA = """{
  "reasoning": "one or two sentences on why you split the work this way",
  "answer_directly": "a complete answer, ONLY when no specialist is needed; otherwise null",
  "steps": [
    {
      "id": "s1",
      "agent": "<one of the agent names above>",
      "objective": "what this agent must produce, specific enough to act on alone",
      "depends_on": []
    }
  ]
}"""


class Planner:
    """The Delivery Lead's planning brain: decide who does what, and whether more is needed.

    It makes two kinds of call, both returning a validated `Plan`: `plan` for a fresh
    request, and `replan` after results are in. It tracks its own `llm_calls` and `usage`
    so the orchestrator can report the full cost of a run, planning included.
    """

    def __init__(self, llm: LLMClient, agents: AgentRegistry, settings: Settings) -> None:
        """Create a planner over a model client, the agent roster and the settings."""
        self._llm = llm
        self._agents = agents
        self._settings = settings
        self.llm_calls = 0
        # Every token this planner spends, including repair attempts and replan
        # rounds. The orchestrator reports it; nobody should have to guess.
        self.usage = Usage()

    # -- prompts -----------------------------------------------------------

    def _system_prompt(self) -> str:
        """The Delivery Lead's standing instructions, with the live roster embedded.

        The delegation rules here do real work. "Empty `depends_on` is the default" is what makes
        the model fan out instead of chaining everything; "name the PR/service in every
        objective" exists because an agent sees only its own objective plus its upstream outputs,
        never the other steps.
        """
        return f"""You are the Delivery Lead at Northwind Payments. Engineers bring you requests
and you decide who does the work. You do not do the work yourself.

## Your team

{self._agents.roster_prompt()}

## How to delegate

- Pick the smallest set of agents that fully answers the request. At most \
{self._settings.max_plan_steps} steps.
- Give each agent a self-contained objective. It cannot see the other steps' objectives, only
  the outputs of the steps it depends on, so name the pull request number, service or repository
  explicitly in every objective.
- Leave `depends_on` empty for anything that can start immediately -- independent steps run in
  parallel, so an empty list is the default and a dependency needs a reason.
- Add a dependency only when a step genuinely needs an earlier step's findings. "Review PR 482"
  and "assess its test coverage" are independent. "Draft release notes" after "check what is
  blocking the release" is a real dependency.
- Do not give two agents the same objective, and do not delegate to an agent whose mandate does
  not cover the work.
- If the request is a simple factual question about how this team works, or is too vague to act
  on, set `answer_directly` and return no steps. Everything that touches code, production data,
  tests or the release goes to a specialist.

## Output

Reply with JSON only, in exactly this shape:

{_PLAN_SCHEMA}"""

    # -- planning ----------------------------------------------------------

    async def plan(self, request: str, bus: EventBus) -> Plan:
        """Turn a request into a validated plan, or a direct answer when no specialist is needed.

        Emits `plan.created`. May use one extra model call to repair an invalid plan.

        Raises:
            PlanValidationError: The model produced an invalid plan twice in a row.
            LLMError: The model provider failed.
        """
        messages = [
            Message(role="system", content=self._system_prompt()),
            Message(role="user", content=f"Request from the team:\n\n{request}"),
        ]
        plan = await self._plan_with_repair(messages)
        bus.emit(
            EventType.PLAN_CREATED,
            agent="orchestrator",
            message=plan.reasoning or "Plan created",
            direct=plan.is_direct,
            steps=[step.model_dump() for step in plan.steps],
        )
        return plan

    async def replan(
        self,
        request: str,
        plan: Plan,
        results: list[StepResult],
        bus: EventBus,
    ) -> Plan:
        """Decide whether the results leave a gap worth one more round of work."""
        summary = "\n\n".join(
            f"### {r.step_id} -- {r.agent} ({r.status})\n"
            f"Objective: {r.objective}\n"
            f"{r.error or r.output}"
            for r in results
        )
        messages = [
            Message(role="system", content=self._system_prompt()),
            Message(
                role="user",
                content=(
                    f"Request from the team:\n\n{request}\n\n"
                    f"You delegated this work and got these results back:\n\n{summary}\n\n"
                    "Is anything still missing that one more round of delegation would fix -- a "
                    "step that failed and is worth retrying differently, or a gap the results "
                    "exposed? If the request is now fully answered, return an empty `steps` "
                    "list. Only add steps that change the final answer. Reply with JSON only."
                ),
            ),
        ]
        revision = await self._plan_with_repair(messages, existing_ids={s.id for s in plan.steps})
        if revision.steps:
            bus.emit(
                EventType.PLAN_REVISED,
                agent="orchestrator",
                message=revision.reasoning or "Follow-up work scheduled",
                steps=[step.model_dump() for step in revision.steps],
            )
        return revision

    # -- internals ---------------------------------------------------------

    async def _plan_with_repair(
        self, messages: list[Message], existing_ids: set[str] | None = None
    ) -> Plan:
        """Ask for a plan and, if it is invalid, give the model exactly one chance to fix it.

        On an invalid plan the conversation is extended with the model's own reply plus the
        validation error, so the second attempt knows precisely what was wrong. Two attempts
        is a deliberate cap: one repair fixes the usual slip (a bad agent name, a missing
        dependency); a model that fails twice is unlikely to succeed on a third try, and each
        try costs a call.

        Args:
            messages: The planning conversation so far.
            existing_ids: Step ids already used in earlier rounds (empty for a first plan). They
                may be depended on but not reused.
        """
        # Temperature 0 and JSON mode below: planning should be repeatable and machine-readable.
        for attempt in (1, 2):
            completion = await self._llm.complete(
                messages,
                model=self._settings.orchestrator_model,
                temperature=0.0,
                json_object=True,
            )
            self.llm_calls += 1
            self.usage += completion.usage
            try:
                plan = self._parse(completion.text)
                self._validate(plan, existing_ids or set())
                return plan
            except PlanValidationError as exc:
                if attempt == 2:
                    raise
                log.warning("planner.invalid_plan", extra={"error": exc.message})
                messages = [
                    *messages,
                    completion.message,
                    Message(
                        role="user",
                        content=(
                            f"That plan was rejected: {exc.message}\n"
                            "Return a corrected plan as JSON only."
                        ),
                    ),
                ]
        # Unreachable: attempt 2 either returns or re-raises. Kept so the function provably returns.
        raise PlanValidationError("Planner did not produce a valid plan.")  # pragma: no cover

    def _parse(self, text: str) -> Plan:
        """Parse the model's text into a `Plan`; any failure becomes a `PlanValidationError`.

        Three distinct failures get distinct messages, because the message is shown to the model
        on the repair attempt: not JSON, JSON that is not an object, and JSON of the wrong shape.
        """
        try:
            payload = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise PlanValidationError(f"Plan was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise PlanValidationError("Plan must be a JSON object.")
        try:
            return Plan.model_validate(payload)
        except ValidationError as exc:
            raise PlanValidationError(f"Plan did not match the schema: {exc}") from exc

    def _validate(self, plan: Plan, existing_ids: set[str]) -> None:
        """Reject plans we must not execute. Checks run cheapest and most fundamental first.

        1. An empty plan is only acceptable as a direct answer - or as a revision, where
           "nothing more to do" is a legitimate result.
        2. The step count is within `max_plan_steps` (a cost guardrail, enforced in code).
        3. Every step names a real agent, has a unique id and a non-empty objective.
        4. Every dependency refers to a step in this plan or an earlier round, and not to itself.
        5. The dependency graph has no cycle.

        Raises:
            PlanValidationError: With a message written for the model to read and act on.
        """
        if not plan.steps:
            if plan.answer_directly and plan.answer_directly.strip():
                return
            # A revision with no steps means "nothing more to do"; only a FIRST plan must have
            # content.
            if existing_ids:  # a revision legitimately returns no steps
                return
            raise PlanValidationError(
                "A plan must contain at least one step, or set `answer_directly`."
            )

        if len(plan.steps) > self._settings.max_plan_steps:
            raise PlanValidationError(
                f"Plan has {len(plan.steps)} steps; the limit is {self._settings.max_plan_steps}."
            )

        known = self._agents.names()
        ids: set[str] = set()
        for step in plan.steps:
            if step.agent not in known:
                raise PlanValidationError(
                    f"Step {step.id!r} delegates to unknown agent {step.agent!r}. "
                    f"Available agents: {', '.join(known)}."
                )
            if step.id in ids or step.id in existing_ids:
                raise PlanValidationError(f"Duplicate step id {step.id!r}.")
            if not step.objective.strip():
                raise PlanValidationError(f"Step {step.id!r} has an empty objective.")
            ids.add(step.id)

        # A step may depend on anything in this plan or on a step that already ran in an earlier
        # round. That is how a follow-up builds on earlier findings.
        resolvable = ids | existing_ids
        for step in plan.steps:
            unknown = [dep for dep in step.depends_on if dep not in resolvable]
            if unknown:
                raise PlanValidationError(
                    f"Step {step.id!r} depends on unknown step(s): {', '.join(unknown)}."
                )
            if step.id in step.depends_on:
                raise PlanValidationError(f"Step {step.id!r} depends on itself.")

        _assert_acyclic(plan)


def _assert_acyclic(plan: Plan) -> None:
    """Kahn's algorithm -- if anything is left over, the graph has a cycle."""
    # step id -> the not-yet-removed dependencies that are inside THIS plan.
    pending = {step.id: set(step.depends_on) & {s.id for s in plan.steps} for step in plan.steps}
    while True:
        # Steps with nothing left to wait for can be taken off the graph.
        ready = [step_id for step_id, deps in pending.items() if not deps]
        # Nothing removable. Either the graph is empty (fine) or what remains is a cycle.
        if not ready:
            break
        for step_id in ready:
            pending.pop(step_id)
            for deps in pending.values():
                deps.discard(step_id)
    if pending:
        raise PlanValidationError(
            f"Plan contains a dependency cycle between: {', '.join(sorted(pending))}."
        )
