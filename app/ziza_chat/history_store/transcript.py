"""Turning stored messages back into a readable conversation.

The transcript is kept in pydantic-ai's message format so a turn can be
replayed to the model exactly as it happened. That format carries more than a
reader wants — tool calls, retrieved passages, the instructions — so this pulls
out the two things a visitor actually saw.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from app.ziza_chat.history_store.summary import (
    is_summary_acknowledgement,
    is_summary_turn,
)


@dataclass(frozen=True)
class TranscriptTurn:
    role: Literal["user", "assistant"]
    text: str
    at: datetime


def to_transcript(
    messages: Sequence[ModelMessage], at: datetime
) -> list[TranscriptTurn]:
    """What the visitor saw, in order.

    Tool calls and their returns are dropped: they are how the answer was
    reached, not part of it, and the panel that shows them is fed from the live
    stream. A rolling summary is dropped too — it is written for the model, and
    reads as something the visitor said when it is not.
    """
    turns: list[TranscriptTurn] = []
    for message in messages:
        if is_summary_turn(message) or is_summary_acknowledgement(message):
            continue
        if isinstance(message, ModelRequest):
            turns.extend(
                TranscriptTurn(role="user", text=part.content, at=at)
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
        elif isinstance(message, ModelResponse):
            turns.extend(
                TranscriptTurn(role="assistant", text=part.content, at=at)
                for part in message.parts
                if isinstance(part, TextPart) and part.content.strip()
            )
    return turns
