from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.ziza_chat.hitl.models import DeferredKind


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class SuggestionKind(str, Enum):
    ADD_PAGE = "add_page"


class Suggestion(BaseModel):
    """Something the visitor may want next, offered alongside a finished answer.

    Not a deferred call: the run completed, the answer stands, and ignoring
    this costs nothing. Acting on one means sending `message` as an ordinary
    chat message, so every kind resolves the same way and a page still enters
    the knowledge base through the one path that asks before indexing.
    """

    kind: SuggestionKind
    label: str
    message: str
    url: str | None = None


class PendingCallRead(BaseModel):
    tool_call_id: str
    tool_name: str
    kind: DeferredKind
    details: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    pending_calls: list[PendingCallRead] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)


class ChatStreamEvent(BaseModel):
    type: str
    chunk: str | None = None
    pending_call: PendingCallRead | None = None
    suggestions: list[Suggestion] | None = None


class ApprovalDecisionRequest(BaseModel):
    session_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    approved: bool


class LinkSelectionRequest(BaseModel):
    session_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    # Empty is a valid answer meaning "just the page itself", not "nothing
    # to do" — walking away from the offer is the separate decline path.
    selected_links: list[str] = Field(default_factory=list, max_length=20)


class KnowledgeIngestResponse(BaseModel):
    session_id: str
    source: str
    chunks_ingested: int
    images_described: int = 0
    pages_summarised: int = 0
    documents_used: int = 0
    documents_allowed: int = 0
    images_failed: int = 0
    searchable: bool


class KnowledgeClearResponse(BaseModel):
    session_id: str
    chunks_deleted: int
