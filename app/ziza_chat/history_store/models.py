from datetime import datetime
from typing import Any

from beanie import Document
from pydantic import Field
from pymongo import IndexModel

from app.core.config import settings
from app.core.utils.time import utc_now


class ConversationTurn(Document):
    session_id: str
    messages: list[dict[str, Any]]
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "conversation_turns"
        indexes = [
            IndexModel(
                [("session_id", 1), ("created_at", 1)],
                name="session_turns",
            ),
            IndexModel(
                [("created_at", 1)],
                name="ttl_conversation_turns",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]


class ConversationSummary(Document):
    session_id: str
    summary: str
    covers_turns: int
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "conversation_summaries"
        indexes = [
            IndexModel(
                [("session_id", 1)],
                unique=True,
                name="uniq_session_summary",
            ),
            IndexModel(
                [("created_at", 1)],
                name="ttl_conversation_summaries",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]
