"""Unit tests for the content-sync layer that don't require a live MongoDB.

End-to-end CRUD + SSE behaviour is covered by a manual/integration check against
a running Mongo replica set (change streams require one).
"""

from app.content_sync.events import format_sse
from app.content_sync.schemas import SyncEvent, SyncFlagRead
from app.core.utils.time import utc_now


def _read(key: str = "deploy", enabled: bool = True) -> SyncFlagRead:
    now = utc_now()
    return SyncFlagRead(
        session_id="sess-1",
        key=key,
        value="ship it",
        enabled=enabled,
        created_at=now,
        updated_at=now,
    )


def test_format_sse_created_event() -> None:
    event = SyncEvent(type="flag.created", key="deploy", flag=_read())
    wire = format_sse(event)

    assert wire.startswith("event: flag.created\n")
    assert "data: " in wire
    assert wire.endswith("\n\n")
    # exclude_none keeps the payload tight
    assert "flags" not in wire


def test_format_sse_snapshot_event() -> None:
    event = SyncEvent(type="flag.snapshot", flags=[_read("a"), _read("b")])
    wire = format_sse(event)

    assert wire.startswith("event: flag.snapshot\n")
    assert '"key": "a"' in wire or '"key":"a"' in wire
