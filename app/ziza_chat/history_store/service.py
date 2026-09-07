import asyncio
import logging
from typing import Sequence

from pydantic_ai.messages import ModelMessage

from app.core.db.db_config import is_db_configured
from app.ziza_chat.agents.conversation_summary import fold_into_summary
from app.ziza_chat.history_store.store import build_refusal_turn, get_history_store
from app.ziza_chat.history_store.summary import render_transcript
from app.ziza_chat.history_store.trimming import turns_to_drop

logger = logging.getLogger(__name__)

_pending_summaries: set[asyncio.Task[None]] = set()


async def load_history(session_id: str) -> list[ModelMessage]:
    if not is_db_configured():
        return []
    return await get_history_store().load(session_id)


async def append_turn(session_id: str, messages: Sequence[ModelMessage]) -> None:
    if not is_db_configured():
        return
    await get_history_store().append(session_id, messages)
    schedule_summary_refresh(session_id)


async def record_refusal(
    session_id: str, visitor_message: str, refusal: str
) -> None:
    await append_turn(session_id, build_refusal_turn(visitor_message, refusal))


async def clear_history(session_id: str) -> int:
    if not is_db_configured():
        return 0
    turns_removed = await get_history_store().clear(session_id)
    if turns_removed:
        logger.info(
            "cleared %d conversation turn(s) for %s", turns_removed, session_id
        )
    return turns_removed


def schedule_summary_refresh(session_id: str) -> None:
    task = asyncio.create_task(refresh_summary(session_id))
    _pending_summaries.add(task)
    task.add_done_callback(_pending_summaries.discard)


async def refresh_summary(session_id: str) -> None:
    try:
        store = get_history_store()
        turns_now_summarised = turns_to_drop(await store.count_turns(session_id))
        if not turns_now_summarised:
            return
        existing = await store.load_summary(session_id)
        already_summarised = existing.covers_turns if existing else 0
        if turns_now_summarised <= already_summarised:
            return
        newly_dropped = await store.load_turn_range(
            session_id,
            skip=already_summarised,
            limit=turns_now_summarised - already_summarised,
        )
        if not newly_dropped:
            return
        folded = await fold_into_summary(
            existing.summary if existing else "", render_transcript(newly_dropped)
        )
        await store.save_summary(
            session_id, folded.to_prompt_text(), turns_now_summarised
        )
        logger.info(
            "summarised turns %d-%d for %s",
            already_summarised + 1,
            turns_now_summarised,
            session_id,
        )
    except Exception as failure:
        logger.warning("Could not refresh the summary for %s: %s", session_id, failure)
