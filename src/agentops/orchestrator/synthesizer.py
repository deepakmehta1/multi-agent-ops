"""Turns a pile of specialist reports into one answer for the person who asked."""

from __future__ import annotations

from agentops.core.config import Settings
from agentops.llm.base import LLMClient, Message, Usage
from agentops.orchestrator.models import StepResult, StepStatus

_SYSTEM = """You are the Delivery Lead at Northwind Payments. You delegated a request to your
specialists and now you answer the person who asked.

Rules:
- Lead with the answer or the decision. The reader wants the conclusion in the first sentence.
- Use only what your specialists reported. Add no facts of your own. If two reports disagree,
  say so rather than quietly picking one.
- Attribute substantive findings to the specialist who made them, in passing: "the on-call
  review found ...", not a formal citation.
- If a step failed or was skipped, say what is therefore unverified. Never present a partial
  answer as complete.
- Keep the specialists' concrete artefacts -- code blocks, release-note drafts, specific
  commands -- intact rather than summarising them away.
- Use markdown. Be as short as the content allows. No preamble, no sign-off."""


class Synthesizer:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings
        self.llm_calls = 0
        self.usage = Usage()

    async def synthesize(self, request: str, results: list[StepResult]) -> tuple[str, Usage]:
        if not results:
            return ("No specialist work was carried out for this request.", Usage())

        succeeded = [r for r in results if r.status is StepStatus.SUCCEEDED]
        if len(succeeded) == 1 and len(results) == 1:
            # A single specialist already wrote the answer; re-writing it only loses detail.
            return (succeeded[0].output, Usage())

        reports = "\n\n---\n\n".join(_render(result) for result in results)
        completion = await self._llm.complete(
            [
                Message(role="system", content=_SYSTEM),
                Message(
                    role="user",
                    content=(
                        f"## The request\n{request}\n\n"
                        f"## What your specialists reported\n\n{reports}\n\n"
                        "Write the answer."
                    ),
                ),
            ],
            model=self._settings.orchestrator_model,
            temperature=0.2,
        )
        self.llm_calls += 1
        self.usage += completion.usage
        return completion.text, completion.usage


def _render(result: StepResult) -> str:
    header = f"### {result.agent} -- {result.objective}"
    if result.status is not StepStatus.SUCCEEDED:
        return f"{header}\n**{result.status.upper()}:** {result.error or 'no output'}"
    body = result.output
    if result.truncated:
        body += "\n\n_(This agent ran out of its tool budget; its report may be incomplete.)_"
    return f"{header}\n{body}"
