import logging
from functools import lru_cache
from typing import Any, Sequence

from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from app.ziza_chat.history_store.models import ConversationSummary, ConversationTurn
from app.ziza_chat.history_store.summary import build_summary_turn
from app.ziza_chat.history_store.trimming import turns_to_drop

logger = logging.getLogger(__name__)


def serialize_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return list(ModelMessagesTypeAdapter.dump_python(list(messages), mode="json"))


def deserialize_turns(
    stored_turns: Sequence[Sequence[dict[str, Any]]],
) -> list[ModelMessage]:
    restored_messages: list[ModelMessage] = []
    for stored_messages in stored_turns:
        try:
            restored_messages.extend(
                ModelMessagesTypeAdapter.validate_python(list(stored_messages))
            )
        except ValidationError as failure:
            logger.warning("Skipped an unreadable conversation turn: %s", failure)
    return restored_messages


def build_refusal_turn(visitor_message: str, refusal: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=visitor_message)]),
        ModelResponse(parts=[TextPart(content=refusal)]),
    ]


class MongoHistoryStore:
    @staticmethod
    async def count_turns(session_id: str) -> int:
        return await ConversationTurn.find(
            ConversationTurn.session_id == session_id
        ).count()

    @staticmethod
    async def load_summary(session_id: str) -> ConversationSummary | None:
        return await ConversationSummary.find_one(
            ConversationSummary.session_id == session_id
        )

    @staticmethod
    async def save_summary(session_id: str, summary: str, covers_turns: int) -> None:
        existing = await ConversationSummary.find_one(
            ConversationSummary.session_id == session_id
        )
        if existing is None:
            await ConversationSummary(
                session_id=session_id, summary=summary, covers_turns=covers_turns
            ).insert()
            return
        existing.summary = summary
        existing.covers_turns = covers_turns
        await existing.save()

    @staticmethod
    async def load_turns(session_id: str, skip: int) -> list[ModelMessage]:
        turns = (
            await ConversationTurn.find(ConversationTurn.session_id == session_id)
            .sort("+created_at")
            .skip(skip)
            .to_list()
        )
        return deserialize_turns([turn.messages for turn in turns])

    @staticmethod
    async def load_turn_range(
        session_id: str, skip: int, limit: int
    ) -> list[ModelMessage]:
        turns = (
            await ConversationTurn.find(ConversationTurn.session_id == session_id)
            .sort("+created_at")
            .skip(skip)
            .limit(limit)
            .to_list()
        )
        return deserialize_turns([turn.messages for turn in turns])

    async def load(self, session_id: str) -> list[ModelMessage]:
        older_turns_to_skip = turns_to_drop(await self.count_turns(session_id))
        window = await self.load_turns(session_id, older_turns_to_skip)
        if not older_turns_to_skip:
            return window
        summary = await self.load_summary(session_id)
        if summary is None:
            return window
        return build_summary_turn(summary.summary) + window

    @staticmethod
    async def append(session_id: str, messages: Sequence[ModelMessage]) -> None:
        if not messages:
            return
        try:
            await ConversationTurn(
                session_id=session_id, messages=serialize_messages(messages)
            ).insert()
        except Exception as failure:
            logger.warning("Could not store a turn for %s: %s", session_id, failure)

    @staticmethod
    async def clear(session_id: str) -> int:
        await ConversationSummary.find(
            ConversationSummary.session_id == session_id
        ).delete()
        deletion = await ConversationTurn.find(
            ConversationTurn.session_id == session_id
        ).delete()
        return int(deletion.deleted_count) if deletion else 0


@lru_cache
def get_history_store() -> MongoHistoryStore:
    return MongoHistoryStore()
