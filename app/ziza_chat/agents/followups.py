"""Proposing the one question worth asking next, or none at all.

Two models, deliberately from different providers. The writer produces
candidates; the grader picks at most one. A model asked to judge questions it
has just written is reluctant to reject them all, and rejecting is the point —
a suggestion that appears after every answer is one nobody reads.
"""

import logging
from functools import lru_cache
from typing import Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.ziza_chat.config import ziza_settings

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 5

MAX_PASSAGE_CHARS = 6_000

WRITER_PROMPT = """\
You write follow-up questions for a visitor reading an answer drawn from their
own uploaded documents.

You are shown the question they asked, the answer they were given, and the
passages that answer came from. Propose questions that those passages, or
material plainly adjacent to them, could answer.

What makes a good one:

- It goes somewhere the answer opened up but did not follow — a named thing
  left unexplained, a figure without its context, a process whose next step is
  described elsewhere.
- It is phrased the way the visitor would type it, not the way a search index
  would store it. Short, direct, no preamble.
- It can be answered from their documents. You are not writing questions for a
  general-purpose assistant, and anything needing outside knowledge is useless
  here.

What to avoid:

- Restating the question just asked, or asking for the same answer in other
  words.
- Questions whose answer is already in the answer they just read.
- Vague openers — "tell me more", "what else", "can you elaborate". They carry
  no information and could follow any answer at all.
- Anything the passages give you no reason to believe is covered.
- Questions the passages merely *contain* rather than answer. A passage quoted
  from a call script, form, or checklist is a list of things nobody knows yet;
  asking one back gets the visitor told to go and ask someone else.

Return up to five, ordered best first. Returning fewer is better than padding,
and returning none is correct when the material genuinely offers nothing worth
asking. An empty list is a real answer, not a failure."""

GRADER_PROMPT = """\
You decide whether any of several proposed follow-up questions is worth putting
in front of a visitor, and if so, which one.

You did not write these. Judge them as a reader would: you are shown the
visitor's question, the answer they received, and the candidates.

Choose one only if it would make the conversation genuinely better — it asks
something a curious reader of that answer would actually want to know next, and
it is specific enough that its answer would differ from the answer already
given.

Reject all of them when they are obvious, when their answers are already
contained in the answer just given, when they are vague enough to follow any
answer at all, or when they restate the original question. Rejecting is the
common case and costs nothing; a weak suggestion is worse than no suggestion,
because it teaches the visitor that these are not worth reading.

Return the chosen question copied exactly as written, or null."""


class FollowUpCandidates(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        max_length=MAX_CANDIDATES,
        description=(
            "Follow-up questions answerable from the visitor's documents, best "
            "first, each phrased as they would type it. Empty when the material "
            "offers nothing worth asking."
        ),
    )


class FollowUpChoice(BaseModel):
    chosen: str | None = Field(
        default=None,
        description=(
            "The one question worth showing, copied exactly from the "
            "candidates, or null when none of them is worth showing."
        ),
    )
    reason: str = Field(
        default="",
        description="One short sentence on why, for the logs.",
    )


@lru_cache
def get_followup_writer() -> Agent[None, FollowUpCandidates]:
    return Agent[None, FollowUpCandidates](
        ziza_settings.ziza_followup_model,
        output_type=FollowUpCandidates,
        instructions=WRITER_PROMPT,
        model_settings=ModelSettings(max_tokens=512),
    )


@lru_cache
def get_followup_grader() -> Agent[None, FollowUpChoice]:
    return Agent[None, FollowUpChoice](
        ziza_settings.ziza_suggestion_grader_model,
        output_type=FollowUpChoice,
        instructions=GRADER_PROMPT,
        model_settings=ModelSettings(temperature=0, max_tokens=512),
    )


def build_context(question: str, answer: str, passages: Sequence[str]) -> str:
    material = "\n\n".join(passages)[:MAX_PASSAGE_CHARS]
    return (
        f"The visitor asked: {question}\n\n"
        f"They were told:\n{answer}\n\n"
        f"Drawn from these passages:\n{material}"
    )


async def propose_followups(
    question: str, answer: str, passages: Sequence[str]
) -> list[str]:
    result = await get_followup_writer().run(
        build_context(question, answer, passages)
    )
    return [text.strip() for text in result.output.questions if text.strip()]


async def choose_followup(
    question: str, answer: str, candidates: Sequence[str]
) -> str | None:
    """The best candidate, or None when none of them earns a place.

    Only a candidate returned verbatim is accepted: a grader that rewrites what
    it was given has stopped grading and started writing, and its output has
    been through neither the writer's instructions nor the retrieval check.
    """
    if not candidates:
        return None
    listed = "\n".join(f"{index}. {text}" for index, text in enumerate(candidates, 1))
    result = await get_followup_grader().run(
        f"{build_context(question, answer, [])}\n\nCandidates:\n{listed}"
    )
    chosen = (result.output.chosen or "").strip()
    if not chosen:
        logger.info("no follow-up worth showing: %s", result.output.reason)
        return None
    if chosen not in candidates:
        logger.warning("grader returned a question it was not given: %r", chosen)
        return None
    return chosen
