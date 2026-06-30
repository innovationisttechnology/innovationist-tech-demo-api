import asyncio
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.content_sync import service
from app.content_sync.connection_manager import connection_manager
from app.content_sync.events import format_sse
from app.content_sync.schemas import (
    SyncEvent,
    SyncFlagCreate,
    SyncFlagRead,
    SyncFlagUpdate,
)

router = APIRouter(prefix="/flags", tags=["content-sync"])


async def get_session_id(
    session_id: str | None = Query(default=None),
    x_session_id: str | None = Header(default=None),
) -> str:
    resolved = session_id or x_session_id
    if not resolved:
        raise HTTPException(
            status_code=422,
            detail="session_id is required (query param or X-Session-Id header)",
        )
    return resolved


SessionId = Annotated[str, Depends(get_session_id)]


@router.get("", response_model=list[SyncFlagRead])
async def list_flags(session_id: SessionId) -> list[SyncFlagRead]:
    return await service.list_flags(session_id)


@router.post("", response_model=SyncFlagRead, status_code=201)
async def create_flag(
    session_id: SessionId, payload: SyncFlagCreate
) -> SyncFlagRead:
    return await service.create_flag(session_id, payload)


@router.get("/stream")
async def stream_flags(session_id: SessionId, request: Request) -> StreamingResponse:
    queue = connection_manager.connect(session_id)

    async def event_generator() -> AsyncIterator[str]:
        try:
            # Send current state immediately so a new client is in sync.
            flags = await service.list_flags(session_id)
            yield format_sse(SyncEvent(type="flag.snapshot", flags=flags))

            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if data is None:
                        break
                    yield data
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            connection_manager.disconnect(session_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{key}", response_model=SyncFlagRead)
async def get_flag(session_id: SessionId, key: str) -> SyncFlagRead:
    return await service.get_flag(session_id, key)


@router.put("/{key}", response_model=SyncFlagRead)
async def update_flag(
    session_id: SessionId, key: str, payload: SyncFlagUpdate
) -> SyncFlagRead:
    return await service.update_flag(session_id, key, payload)


@router.delete("/{key}", status_code=204)
async def delete_flag(session_id: SessionId, key: str) -> None:
    await service.delete_flag(session_id, key)
