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
    this costs nothing. `kind` is what the client routes on — add_page posts
    `url` to /knowledge/url/links with `source_url` as the page it came from.
    """

    kind: SuggestionKind
    label: str
    url: str | None = None
    source_url: str | None = None


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


class KnowledgeIngestRequest(BaseModel):
    session_id: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=100_000)


class CandidateLinkRead(BaseModel):
    url: str
    text: str


class KnowledgeUrlRequest(BaseModel):
    session_id: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=2048)


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
    # Same-origin pages the ingested page links to, offered so the visitor can
    # pull in the rest of a doc site. Empty for anything but the page they
    # submitted themselves.
    candidate_links: list[CandidateLinkRead] = Field(default_factory=list)


class UrlLinkSelectionRequest(BaseModel):
    session_id: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=2048)
    selected_links: list[str] = Field(min_length=1, max_length=20)


class FailedLink(BaseModel):
    url: str
    reason: str


class UrlLinkSelectionResponse(BaseModel):
    session_id: str
    indexed: list[KnowledgeIngestResponse] = Field(default_factory=list)
    failed: list[FailedLink] = Field(default_factory=list)
    documents_used: int = 0
    documents_allowed: int = 0


class KnowledgeClearResponse(BaseModel):
    session_id: str
    chunks_deleted: int
