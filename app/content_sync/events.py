import json

from app.content_sync.schemas import SyncEvent


def format_sse(event: SyncEvent) -> str:
    """Serialize an event into the SSE wire format (`event:` + `data:`)."""
    data = json.dumps(event.model_dump(mode="json", exclude_none=True))
    return f"event: {event.type}\ndata: {data}\n\n"
