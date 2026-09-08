import logging
from typing import Sequence

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelMessage

from app.core.db.db_config import is_db_configured
from app.ziza_chat.history_store.service import append_turn
from app.ziza_chat.history_store.store import (
    build_declined_turn,
    deserialize_turns,
    serialize_messages,
)
from app.ziza_chat.hitl.models import PendingApproval
from app.ziza_chat.hitl.store import get_approval_store

logger = logging.getLogger(__name__)


class NoPendingApprovalError(Exception):
    pass


async def record_pending(
    session_id: str,
    requests: DeferredToolRequests,
    intent: str,
    messages: Sequence[ModelMessage],
) -> PendingApproval | None:
    if not is_db_configured() or not requests.approvals:
        return None
    call = requests.approvals[0]
    details = requests.metadata.get(call.tool_call_id) or {}
    pending = await get_approval_store().replace(
        session_id=session_id,
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        intent=intent,
        details=details,
        messages=serialize_messages(messages),
    )
    logger.info(
        "awaiting approval for %s on %s (call %s)",
        call.tool_name,
        session_id,
        call.tool_call_id,
    )
    return pending


async def find_pending(session_id: str) -> PendingApproval | None:
    if not is_db_configured():
        return None
    return await get_approval_store().find(session_id)


async def claim_pending(session_id: str, tool_call_id: str) -> PendingApproval:
    pending = await find_pending(session_id)
    if pending is None or pending.tool_call_id != tool_call_id:
        raise NoPendingApprovalError(
            "No approval is pending for this session and tool call."
        )
    await get_approval_store().discard(session_id)
    return pending


ABANDONED_DECLINE = (
    "The visitor moved on without confirming, so this was not carried out."
)


async def decline_abandoned(session_id: str) -> bool:
    """Treat a still-pending approval as declined.

    Called when the visitor sends another message instead of answering: the
    offer is over, and leaving the record to expire would hide that the action
    was never taken.
    """
    pending = await find_pending(session_id)
    if pending is None:
        return False
    declined_turn = build_declined_turn(
        deserialize_turns([pending.messages]), ABANDONED_DECLINE
    )
    if declined_turn:
        await append_turn(session_id, declined_turn)
    await get_approval_store().discard(session_id)
    logger.info(
        "declined an abandoned approval for %s (call %s)",
        session_id,
        pending.tool_call_id,
    )
    return True


async def discard_pending(session_id: str) -> None:
    if not is_db_configured():
        return
    await get_approval_store().discard(session_id)
