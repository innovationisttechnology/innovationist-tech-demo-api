from functools import lru_cache
from typing import Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.ziza_chat.config import ziza_settings

MAX_STARTER_QUESTIONS = 3

STARTER_PROMPT = """\
You write the first questions someone would ask about a document they have just
added to a search tool, having not read it.

They know what the document is called and nothing else. Your questions are the
fastest route from that to knowing whether it holds what they came for, so make
each one open a different part of it rather than three angles on the same
paragraph.

What makes a good one:

- It names something specific the document actually contains — a policy, a
  figure, a person, a process — so the answer is concrete rather than a summary
  of the summary.
- It is phrased the way the visitor would type it. Short, direct, no preamble.
- It can be answered from this document alone.

What to avoid:

- "What is this document about?" and every variant of it. They already know
  roughly what it is; that question wastes the slot.
- Questions whose answer is the document's title or heading restated.
- Anything needing knowledge from outside the text you were given.
- Three questions that would all be answered by the same passage.

Watch for documents that are themselves made of questions — a call script, an
interview guide, a form, a questionnaire, an agenda, a checklist of things to
find out. The questions written in them are the open ones, the whole reason the
document exists; it does not answer them and neither can you. Asking one back
produces nothing but a restatement that the document says to go and ask
someone. Ask instead about what such a document *does* state: who to contact,
what to bring, what order to do things in, what to record, what it says to do
if the answer is no.

Return at most three, best first. Fewer is better than padding. If the text is
an error page, a login wall, boilerplate, or too thin to hold a specific
question, return none — an empty list is a real answer."""


GRADER_PROMPT = """\
You decide which proposed opening questions a document actually answers.

You are shown the document and a list of candidates. Keep a candidate only if
someone reading this document could come away with the answer. Copy the ones
you keep exactly as written; invent nothing.

Apply one test to each candidate: if the visitor asked it, what could you say
back using only this document? Find the words you would quote. If the best you
could do is report that the document raises the same question, tells the reader
to go and find out, or leaves a blank to fill in later, drop it — that is not
an answer, and the visitor gets nothing from asking.

This is the failure worth catching, because it disguises itself well. Scripts,
forms, interview guides, checklists and agendas are built out of open
questions; the document records them precisely because nobody knows the answers
yet. A candidate matching one of those lines looks like the best possible fit —
it is nearly word-for-word what the document says — and is worthless.

Keeping none is a normal outcome and costs nothing. A document that only asks
questions should keep none of them, however well they match."""


class AnsweredQuestions(BaseModel):
    answered: list[str] = Field(
        default_factory=list,
        description=(
            "The candidates this document actually answers, copied exactly. "
            "Empty when it answers none of them."
        ),
    )


class StarterQuestionList(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        max_length=MAX_STARTER_QUESTIONS,
        description=(
            "Opening questions answerable from this document, best first, each "
            "phrased as the visitor would type it and each opening a different "
            "part of the text. Empty when the document holds nothing specific "
            "enough to ask about."
        ),
    )


@lru_cache
def get_starter_question_agent() -> Agent[None, StarterQuestionList]:
    return Agent[None, StarterQuestionList](
        ziza_settings.ziza_starter_question_model,
        output_type=StarterQuestionList,
        instructions=STARTER_PROMPT,
        model_settings=ModelSettings(max_tokens=512),
    )


@lru_cache
def get_starter_grader() -> Agent[None, AnsweredQuestions]:
    return Agent[None, AnsweredQuestions](
        ziza_settings.ziza_suggestion_grader_model,
        output_type=AnsweredQuestions,
        instructions=GRADER_PROMPT,
        model_settings=ModelSettings(temperature=0, max_tokens=512),
    )


async def keep_answered(
    document: str, text: str, candidates: Sequence[str]
) -> list[str]:
    """The candidates the document actually answers.

    Retrieval cannot make this call: it measures whether text similar to the
    question sits in the index, and a question quoted in a call script is the
    closest possible match to itself while being the one thing that script
    cannot answer. Only reading the material settles it, and a grader that did
    not write the candidates is readier to discard them.
    """
    if not candidates:
        return []
    listed = "\n".join(
        f"{index}. {candidate}" for index, candidate in enumerate(candidates, 1)
    )
    result = await get_starter_grader().run(
        f"Document: {document}\n\n"
        f"{text[: ziza_settings.max_summary_input_chars]}\n\n"
        f"Candidates:\n{listed}"
    )
    return [
        answered
        for answered in (kept.strip() for kept in result.output.answered)
        if answered in candidates
    ]


async def propose_starter_questions(document: str, text: str) -> list[str]:
    result = await get_starter_question_agent().run(
        f"Document: {document}\n\n{text[: ziza_settings.max_summary_input_chars]}"
    )
    return [question.strip() for question in result.output.questions if question.strip()]
