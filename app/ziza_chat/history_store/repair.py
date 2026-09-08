import logging
from dataclasses import replace
from typing import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from app.ziza_chat.history_store.summary import is_summary_turn

logger = logging.getLogger(__name__)


def resolved_tool_call_ids(messages: Sequence[ModelMessage]) -> set[str]:
    return {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


def drop_unresolved_tool_calls(
    messages: Sequence[ModelMessage],
) -> list[ModelMessage]:
    resolved = resolved_tool_call_ids(messages)
    repaired: list[ModelMessage] = []
    dropped = 0
    for message in messages:
        if not isinstance(message, ModelResponse):
            repaired.append(message)
            continue
        kept_parts = [
            part
            for part in message.parts
            if not (isinstance(part, ToolCallPart) and part.tool_call_id not in resolved)
        ]
        dropped += len(message.parts) - len(kept_parts)
        if not kept_parts:
            continue
        repaired.append(
            message if len(kept_parts) == len(message.parts) else replace(
                message, parts=kept_parts
            )
        )
    if dropped:
        logger.warning(
            "Dropped %d unanswered tool call(s) from replayed history", dropped
        )
    return repaired


def last_visitor_message(messages: Sequence[ModelMessage]) -> str | None:
    """The most recent thing the visitor typed.

    Retrieved passages and the rolling summary are derived from uploaded
    documents, so they are excluded: this is the only history the classifier
    sees, and letting document text reach it would reopen the injection path
    the scope gate closes.
    """
    for message in reversed(list(messages)):
        if not isinstance(message, ModelRequest) or is_summary_turn(message):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                return part.content
    return None
