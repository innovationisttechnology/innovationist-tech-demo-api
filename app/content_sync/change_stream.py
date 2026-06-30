import asyncio
import logging

from app.content_sync.connection_manager import ConnectionManager
from app.content_sync.events import format_sse
from app.content_sync.models import SyncFlag
from app.content_sync.schemas import SyncEvent, SyncFlagRead

logger = logging.getLogger(__name__)


def _event_type(operation: str, doc: dict[str, object]) -> str:
    if doc.get("deleted"):
        return "flag.deleted"
    if operation == "insert":
        return "flag.created"
    return "flag.updated"


async def watch_sync_flags(manager: ConnectionManager) -> None:
    """Tail the MongoDB change stream and fan out every change to the SSE
    clients of the owning session. Deletes are soft (a `deleted` tombstone),
    so they arrive as updates carrying the full document — session_id and key
    included — and are mapped to flag.deleted here.
    """
    collection = SyncFlag.get_pymongo_collection()

    while True:
        try:
            async with await collection.watch(
                [
                    {
                        "$match": {
                            "operationType": {"$in": ["insert", "update", "replace"]}
                        }
                    }
                ],
                full_document="updateLookup",
            ) as stream:
                async for change in stream:
                    doc = change.get("fullDocument")
                    if doc is None:
                        continue

                    event_type = _event_type(change["operationType"], doc)
                    flag = SyncFlagRead(
                        session_id=doc["session_id"],
                        key=doc["key"],
                        value=doc["value"],
                        enabled=doc["enabled"],
                        created_at=doc["created_at"],
                        updated_at=doc["updated_at"],
                    )
                    event = SyncEvent(type=event_type, key=flag.key, flag=flag)
                    await manager.broadcast(doc["session_id"], format_sse(event))
        except Exception:
            logger.exception("Sync flag change stream error — restarting")
            await asyncio.sleep(5)
