from datetime import datetime
from enum import Enum
from typing import Any

from beanie import Document
from pydantic import BaseModel, Field
from pymongo import IndexModel

from app.core.config import settings
from app.core.utils.time import utc_now


class DeferredKind(str, Enum):
    """Why a tool call was handed back instead of run.

    The two behave differently on resume: an approved call re-executes the tool
    body, while an external call never does — the result supplied from outside
    *is* the tool's return value.
    """

    APPROVAL = "approval"
    CALL = "call"


class CallResolution(BaseModel):
    approved: bool | None = None
    message: str | None = None
    content: str | None = None


class DeferredCall(BaseModel):
    tool_call_id: str
    tool_name: str
    kind: DeferredKind
    # Kept so the original DeferredToolRequests can be rebuilt exactly and
    # build_results can validate against it, rather than trusting a call id
    # off the wire.
    part: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolution: CallResolution | None = None


class PausedRun(Document):
    """A run that stopped mid-turn holding one or more unanswered tool calls.

    One document per session, not per call: a single response can defer several
    calls that resolve separately and out of order, and all of them share the
    one set of paused messages the run has to be resumed with.
    """

    session_id: str
    intent: str
    calls: list[DeferredCall] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "paused_runs"
        indexes = [
            IndexModel(
                [("session_id", 1)],
                unique=True,
                name="uniq_session_paused_run",
            ),
            IndexModel(
                [("created_at", 1)],
                name="ttl_paused_runs",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]
