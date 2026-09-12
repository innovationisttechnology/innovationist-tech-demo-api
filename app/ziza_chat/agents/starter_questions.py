from functools import lru_cache

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

Return at most three, best first. Fewer is better than padding. If the text is
an error page, a login wall, boilerplate, or too thin to hold a specific
question, return none — an empty list is a real answer."""


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


async def propose_starter_questions(document: str, text: str) -> list[str]:
    result = await get_starter_question_agent().run(
        f"Document: {document}\n\n{text[: ziza_settings.max_summary_input_chars]}"
    )
    return [question.strip() for question in result.output.questions if question.strip()]
