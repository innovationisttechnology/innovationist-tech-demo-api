import asyncio
import logging
from typing import Any

from pydantic import ValidationError
from pymongo.errors import PyMongoError

from app.content_sync.connection_manager import ConnectionManager
from app.content_sync.events import format_sse
from app.content_sync.models import SyncFlag
from app.content_sync.schemas import SyncEvent, SyncFlagRead

logger = logging.getLogger(__name__)

RESTART_DELAY_SECONDS = 5


def _event_type(operation: str, doc: dict[str, Any]) -> str:
    if doc.get("deleted"):
        return "flag.deleted"
    if operation == "insert":
        return "flag.created"
    return "flag.updated"


def _to_event(operation: str, doc: dict[str, Any]) -> tuple[str, SyncEvent]:
    """Build the event and the session it belongs to. Raises KeyError or
    ValidationError if the document doesn't match the schema."""
    flag = SyncFlagRead(
        session_id=doc["session_id"],
        key=doc["key"],
        value=doc["value"],
        enabled=doc["enabled"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )
    event = SyncEvent(type=_event_type(operation, doc), key=flag.key, flag=flag)
    return flag.session_id, event


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

                    try:
                        session_id, event = _to_event(change["operationType"], doc)
                    except (KeyError, ValidationError):
                        logger.exception(
                            "Skipping malformed sync flag document %s",
                            doc.get("_id"),
                        )
                        continue

                    await manager.broadcast(session_id, format_sse(event))
        except PyMongoError:
            # Connection drops and change-stream invalidation are the transient
            # failures this loop exists to survive.
            logger.exception("Sync flag change stream error — restarting")
            await asyncio.sleep(RESTART_DELAY_SECONDS)
