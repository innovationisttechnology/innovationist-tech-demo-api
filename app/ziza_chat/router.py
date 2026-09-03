import logging
from typing import Annotated, AsyncIterator

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.db.db_config import is_db_configured
from app.ziza_chat import service
from app.ziza_chat.document_loaders import UnsupportedDocumentError
from app.ziza_chat.document_loaders.web import UnsafeUrlError
from app.ziza_chat.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeClearResponse,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeUrlRequest,
)
from app.ziza_chat.utils import format_sse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ziza", tags=["ziza-ai"])


def require_db() -> None:
    if not is_db_configured():
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured; the knowledge base is unavailable",
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest) -> ChatResponse:
    return await service.chat(body)


@router.post("/chat/stream")
async def chat_stream_endpoint(body: ChatRequest) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        async for chunk in service.stream_chat(body):
            yield format_sse(chunk)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/knowledge", response_model=KnowledgeIngestResponse, status_code=201
)
async def ingest_knowledge_endpoint(
    body: KnowledgeIngestRequest,
) -> KnowledgeIngestResponse:
    require_db()
    return await service.ingest_knowledge(body)


MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post(
    "/knowledge/file", response_model=KnowledgeIngestResponse, status_code=201
)
async def ingest_file_endpoint(
    session_id: Annotated[str, Form(min_length=1)],
    file: Annotated[UploadFile, File()],
) -> KnowledgeIngestResponse:
    require_db()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
        )
    try:
        return await service.ingest_file(
            session_id=session_id,
            filename=file.filename or "upload",
            content_type=file.content_type,
            data=data,
        )
    except UnsupportedDocumentError as unsupported:
        raise HTTPException(status_code=415, detail=str(unsupported)) from unsupported


@router.post("/knowledge/url", response_model=KnowledgeIngestResponse, status_code=201)
async def ingest_url_endpoint(body: KnowledgeUrlRequest) -> KnowledgeIngestResponse:
    require_db()
    try:
        return await service.ingest_url(body.session_id, body.url)
    except UnsafeUrlError as unsafe:
        raise HTTPException(status_code=422, detail=str(unsafe)) from unsafe
    except UnsupportedDocumentError as unsupported:
        raise HTTPException(status_code=415, detail=str(unsupported)) from unsupported
    except httpx.HTTPError as network_failure:
        logger.warning("Fetch failed for %s: %s", body.url, network_failure)
        raise HTTPException(
            status_code=502, detail=f"Could not reach {body.url}."
        ) from network_failure


@router.delete("/knowledge/{session_id}", response_model=KnowledgeClearResponse)
async def clear_knowledge_endpoint(session_id: str) -> KnowledgeClearResponse:
    require_db()
    chunks_deleted = await service.clear_knowledge(session_id)
    return KnowledgeClearResponse(
        session_id=session_id, chunks_deleted=chunks_deleted
    )
