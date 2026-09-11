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


def drop_orphan_tool_returns(
    messages: Sequence[ModelMessage],
) -> list[ModelMessage]:
    """Remove tool results that no preceding tool call accounts for.

    The mirror of drop_unresolved_tool_calls, and the half that bricks a
    session rather than degrading it: Anthropic requires every tool result to
    answer a call in the message directly before it, so one stray result makes
    every later request in that session a 400. Repairing on load is what lets a
    transcript already holding one recover instead of staying poisoned until
    its TTL expires.
    """
    offered: set[str] = set()
    answered: set[str] = set()
    repaired: list[ModelMessage] = []
    dropped = 0

    for message in messages:
        if isinstance(message, ModelResponse):
            offered.update(
                part.tool_call_id
                for part in message.parts
                if isinstance(part, ToolCallPart)
            )
            repaired.append(message)
            continue

        kept_parts = []
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                if part.tool_call_id not in offered or part.tool_call_id in answered:
                    dropped += 1
                    continue
                answered.add(part.tool_call_id)
            kept_parts.append(part)

        if not kept_parts:
            continue
        repaired.append(
            message
            if len(kept_parts) == len(message.parts)
            else replace(message, parts=kept_parts)
        )

    if dropped:
        logger.warning(
            "Dropped %d unmatched tool result(s) from replayed history", dropped
        )
    return repaired


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
