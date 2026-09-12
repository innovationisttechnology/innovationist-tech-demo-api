import json

from app.ziza_chat.schemas import ChatStreamEvent


def format_sse(event: ChatStreamEvent) -> str:
    """Serialize an event into the SSE wire format (`event:` + `data:`)."""
    data = json.dumps(event.model_dump(mode="json", exclude_none=True))
    return f"event: {event.type}\ndata: {data}\n\n"
