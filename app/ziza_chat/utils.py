import json


def format_sse(chunk: str) -> str:
    """Wrap a token chunk as a Server-Sent Events `data:` frame."""
    return f"data: {json.dumps({'chunk': chunk})}\n\n"
