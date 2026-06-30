from datetime import datetime

from beanie import Document
from pydantic import Field
from pymongo import IndexModel

from app.core.utils.time import utc_now


class SyncFlag(Document):
    """A piece of content + a feature toggle, scoped to a test session.

    `value` is the content; `enabled` is the feature switch. Each test session
    (anonymous user trying the demo) owns its own set of flags, keyed by `key`.
    """

    session_id: str
    key: str
    value: str
    enabled: bool = True
    deleted: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "sync_flags"
        indexes = [
            # Unique only among live flags, so a tester can re-add a key they
            # previously soft-deleted (the tombstone is excluded from the index).
            IndexModel(
                [("session_id", 1), ("key", 1)],
                unique=True,
                name="uniq_active_session_key",
                partialFilterExpression={"deleted": False},
            ),
        ]
