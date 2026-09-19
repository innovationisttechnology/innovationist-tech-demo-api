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
    # Optional because sessions that stored their questions before headings
    # existed are still live until their TTL expires, and a required field
    # would make every one of them unreadable.
    category: str | None = None


class SessionStarters(Document):
    """Where a session is in its cold start.

    One document per session rather than per uploaded file, because only one
    set is ever offered and it is offered only once. `conversation_started` is
    the end of it: a visitor who has asked something has found their own way in
    and does not need a way in offered to them.

    Kept rather than held in memory so the visitor who uploads and then reloads
    before asking anything — exactly the person these are for — still has them.
    """

    session_id: str
    document: str | None = None
    questions: list[StoredQuestion] = Field(default_factory=list)
    conversation_started: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "session_starters"
        indexes = [
            IndexModel(
                [("session_id", 1)],
                unique=True,
                name="uniq_session_starters",
            ),
            IndexModel(
                [("created_at", 1)],
                name="ttl_session_starters",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]


async def load_starters(session_id: str) -> SessionStarters | None:
    return await SessionStarters.find_one(SessionStarters.session_id == session_id)


async def starters_are_wanted(session_id: str) -> bool:
    """False once the visitor has asked anything.

    Checked before the questions are written, so a session already in
    conversation costs no model call on its next upload.
    """
    existing = await load_starters(session_id)
    return existing is None or not existing.conversation_started


async def store_starter_questions(
    session_id: str, document: str, questions: Sequence[StoredQuestion]
) -> None:
    try:
        existing = await load_starters(session_id)
        if existing is None:
            await SessionStarters(
                session_id=session_id, document=document, questions=list(questions)
            ).insert()
            return
        if existing.conversation_started:
            return
        existing.document = document
        existing.questions = list(questions)
        existing.created_at = utc_now()
        await existing.save()
    except Exception as failure:
        logger.warning(
            "Could not store starter questions for %s: %s", document, failure
        )


async def mark_conversation_started(session_id: str) -> None:
    """Retire the offer, permanently for this session.

    The questions go with the flag: nothing reads them again, and keeping the
    visitor's own material around after it has stopped being useful is not
    something this demo does.
    """
    try:
        existing = await load_starters(session_id)
        if existing is None:
            await SessionStarters(
                session_id=session_id, conversation_started=True
            ).insert()
            return
        if existing.conversation_started:
            return
        existing.conversation_started = True
        existing.document = None
        existing.questions = []
        await existing.save()
    except Exception as failure:
        logger.warning("Could not retire starters for %s: %s", session_id, failure)


async def clear_starter_questions(session_id: str) -> int:
    result = await SessionStarters.find(
        SessionStarters.session_id == session_id
    ).delete()
    return int(result.deleted_count) if result else 0
