import logging
from datetime import datetime
from typing import Sequence

from beanie import Document
from pydantic import BaseModel, Field
from pymongo import IndexModel

from app.core.config import settings
from app.core.utils.time import utc_now

logger = logging.getLogger(__name__)


class StoredQuestion(BaseModel):
    label: str
    message: str


class StarterQuestions(Document):
    """The opening questions offered for one document.

    Kept so they survive a reload: the visitor who uploads a file and then
    refreshes before asking anything is exactly the person these are for, and
    regenerating would cost another model call for an answer already known.
    """

    session_id: str
    document: str
    questions: list[StoredQuestion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "starter_questions"
        indexes = [
            IndexModel(
                [("session_id", 1), ("document", 1)],
                unique=True,
                name="uniq_session_document_questions",
            ),
            IndexModel(
                [("created_at", 1)],
                name="ttl_starter_questions",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]


async def store_starter_questions(
    session_id: str, document: str, questions: Sequence[StoredQuestion]
) -> None:
    try:
        existing = await StarterQuestions.find_one(
            StarterQuestions.session_id == session_id,
            StarterQuestions.document == document,
        )
        if existing is None:
            await StarterQuestions(
                session_id=session_id, document=document, questions=list(questions)
            ).insert()
            return
        existing.questions = list(questions)
        existing.created_at = utc_now()
        await existing.save()
    except Exception as failure:
        logger.warning("Could not store starter questions for %s: %s", document, failure)


async def load_latest_starter_questions(session_id: str) -> StarterQuestions | None:
    """The most recently added document's questions, not every document's.

    One visible set at a time: a visitor who has added three files is offered
    three questions about the newest, rather than nine about all of them.
    """
    return (
        await StarterQuestions.find(StarterQuestions.session_id == session_id)
        .sort("-created_at")
        .first_or_none()
    )


async def clear_starter_questions(session_id: str) -> int:
    result = await StarterQuestions.find(
        StarterQuestions.session_id == session_id
    ).delete()
    return int(result.deleted_count) if result else 0
