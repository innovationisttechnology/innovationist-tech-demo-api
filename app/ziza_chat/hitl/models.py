from datetime import datetime
from typing import Any

from beanie import Document
from pydantic import Field
from pymongo import IndexModel

from app.core.config import settings
from app.core.utils.time import utc_now


class PendingApproval(Document):
    session_id: str
    tool_call_id: str
    tool_name: str
    intent: str
    details: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "pending_approvals"
        indexes = [
            IndexModel(
                [("session_id", 1)],
                unique=True,
                name="uniq_session_pending_approval",
            ),
            IndexModel(
                [("created_at", 1)],
                name="ttl_pending_approvals",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]
