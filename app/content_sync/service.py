from fastapi import HTTPException

from app.content_sync import repository as repo
from app.content_sync.models import SyncFlag
from app.content_sync.schemas import SyncFlagCreate, SyncFlagRead, SyncFlagUpdate


def _to_read(flag: SyncFlag) -> SyncFlagRead:
    return SyncFlagRead.model_validate(flag)


async def list_flags(session_id: str) -> list[SyncFlagRead]:
    flags = await repo.list_flags(session_id)
    return [_to_read(flag) for flag in flags]


async def get_flag(session_id: str, key: str) -> SyncFlagRead:
    return _to_read(await repo.get_flag_or_raise(session_id, key))


async def create_flag(session_id: str, payload: SyncFlagCreate) -> SyncFlagRead:
    if await repo.find_active_flag(session_id, payload.key) is not None:
        raise HTTPException(status_code=409, detail="Flag key already exists")
    flag = await repo.create_flag(
        session_id, payload.key, payload.value, payload.enabled
    )
    # All broadcasts flow through the change stream (insert -> flag.created).
    return _to_read(flag)


async def update_flag(
    session_id: str, key: str, payload: SyncFlagUpdate
) -> SyncFlagRead:
    flag = await repo.get_flag_or_raise(session_id, key)
    updated = await repo.update_flag(flag, payload.value, payload.enabled)
    # change stream: update -> flag.updated.
    return _to_read(updated)


async def delete_flag(session_id: str, key: str) -> None:
    flag = await repo.get_flag_or_raise(session_id, key)
    # Soft delete: the resulting update carries the full document, so the
    # change stream emits flag.deleted with session_id + key intact.
    await repo.soft_delete_flag(flag)
