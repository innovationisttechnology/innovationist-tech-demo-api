from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    QUESTION = "question"
    COMPLAINT = "complaint"
    CASUAL = "casual conversation"
    TASK_REQUEST = "task request"
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    FOLLOW_UP = "follow-up"
    IDENTITY = "identity"


class Scope(str, Enum):
    KNOWLEDGE_BASE = "knowledge base"
    ASSISTANT = "assistant"
    OUT_OF_SCOPE = "out of scope"


class ClassifyResult(BaseModel):
    scope: Scope = Field(
        default=Scope.KNOWLEDGE_BASE,
        description=(
            "Where an answer would have to come from. "
            "'knowledge base' if answering means reading the visitor's uploaded "
            "documents — any question about a specific topic, person, entity, "
            "or document, whether or not it happens to be uploaded yet. "
            "'assistant' if the message is about this assistant or demo itself, "
            "or is purely social (greetings, thanks, small talk). "
            "'out of scope' if answering would mean drawing on general world "
            "knowledge, doing arithmetic, translating, writing creative or code "
            "content, or giving advice and opinions — anything a general-purpose "
            "assistant does that does not involve reading the visitor's documents."
        ),
    )
    intents: list[Intent] = Field(
        min_length=1,
        description="All intents that apply, ordered from most to least dominant.",
    )
    needs_rag: bool = Field(
        description=(
            "True if any part of the message contains a question, topic request, "
            "or anything that would benefit from retrieved knowledge. "
            "False only for pure social messages."
        )
    )
    rag_query: str | None = Field(
        default=None,
        description=(
            "The core topic or entity to search for (a name, subject, or concept) "
            "if needs_rag is true, otherwise null."
        ),
    )
    rag_ambiguous: bool = Field(
        default=False,
        description=(
            "True if rag_query is vague, partial, or could plausibly match "
            "multiple different things (e.g. a first name only)."
        ),
    )

    @property
    def primary(self) -> Intent:
        return self.intents[0] if self.intents else Intent.QUESTION