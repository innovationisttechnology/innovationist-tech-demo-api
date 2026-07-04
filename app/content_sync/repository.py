from typing import Optional

from beanie.operators import Eq

from app.content_sync.models import SyncFlag
from app.core.utils.repo import find_one_or_raise
from app.core.utils.time import utc_now


async def list_flags(session_id: str) -> list[SyncFlag]:
    return await SyncFlag.find(
        SyncFlag.session_id == session_id, Eq(SyncFlag.deleted, False)
    ).to_list()


async def get_flag_or_raise(session_id: str, key: str) -> SyncFlag:
    return await find_one_or_raise(
        SyncFlag,
        SyncFlag.session_id == session_id,
        SyncFlag.key == key,
        Eq(SyncFlag.deleted, False),
        detail="Flag not found",
    )


async def find_active_flag(session_id: str, key: str) -> Optional[SyncFlag]:
    return await SyncFlag.find_one(
        SyncFlag.session_id == session_id,
        SyncFlag.key == key,
        Eq(SyncFlag.deleted, False),
    )


async def create_flag(
    session_id: str, key: str, value: str, enabled: bool
) -> SyncFlag:
    flag = SyncFlag(session_id=session_id, key=key, value=value, enabled=enabled)
    await flag.insert()
    return flag


async def update_flag(
    flag: SyncFlag, value: Optional[str], enabled: Optional[bool]
) -> SyncFlag:
    if value is not None:
        flag.value = value
    if enabled is not None:
        flag.enabled = enabled
    flag.updated_at = utc_now()
    await flag.save()
    return flag


async def soft_delete_flag(flag: SyncFlag) -> SyncFlag:
    now = utc_now()
    flag.deleted = True
    flag.updated_at = now
    flag.deleted_at = now
    await flag.save()
    return flag
