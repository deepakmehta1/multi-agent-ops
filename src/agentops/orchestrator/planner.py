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
    def __init__(self, llm: LLMClient, agents: AgentRegistry, settings: Settings) -> None:
        self._llm = llm
        self._agents = agents
        self._settings = settings
        self.llm_calls = 0
        # Every token this planner spends, including repair attempts and replan
        # rounds. The orchestrator reports it; nobody should have to guess.
        self.usage = Usage()

    # -- prompts -----------------------------------------------------------

    def _system_prompt(self) -> str:
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
        raise PlanValidationError("Planner did not produce a valid plan.")  # pragma: no cover

    def _parse(self, text: str) -> Plan:
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
        if not plan.steps:
            if plan.answer_directly and plan.answer_directly.strip():
                return
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
    pending = {step.id: set(step.depends_on) & {s.id for s in plan.steps} for step in plan.steps}
    while True:
        ready = [step_id for step_id, deps in pending.items() if not deps]
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
