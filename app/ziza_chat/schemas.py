from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class PendingApprovalRead(BaseModel):
    tool_call_id: str
    tool_name: str
    details: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    pending_approval: PendingApprovalRead | None = None


class ChatStreamEvent(BaseModel):
    type: str
    chunk: str | None = None
    pending_approval: PendingApprovalRead | None = None


class ApprovalDecisionRequest(BaseModel):
    session_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    approved: bool


class KnowledgeIngestRequest(BaseModel):
    session_id: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=100_000)


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


class KnowledgeClearResponse(BaseModel):
    session_id: str
    chunks_deleted: int
