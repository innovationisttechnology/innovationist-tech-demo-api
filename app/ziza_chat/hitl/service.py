from typing import Any, Sequence

from pydantic import TypeAdapter
from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.tools import DeferredToolApprovalResult

from app.core.db.db_config import is_db_configured
from app.ziza_chat.history_store.service import append_turn
from app.ziza_chat.history_store.store import (
    build_declined_turn,
    deserialize_turns,
    serialize_messages,
)
from app.ziza_chat.hitl.models import (
    CallResolution,
    DeferredCall,
    DeferredKind,
    PausedRun,
)
from app.ziza_chat.hitl.store import get_paused_run_store
from app.ziza_chat.tool_logging import log_step

TOOL_CALL_PART_ADAPTER: TypeAdapter[ToolCallPart] = TypeAdapter(ToolCallPart)

DEFAULT_DENIAL = "The visitor declined this action, so nothing was changed."
MISSING_RESULT = "This produced no result."


class UnknownDeferredCallError(Exception):
    pass


def to_deferred_calls(requests: DeferredToolRequests) -> list[DeferredCall]:
    def build(part: ToolCallPart, kind: DeferredKind) -> DeferredCall:
        return DeferredCall(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            kind=kind,
            part=TOOL_CALL_PART_ADAPTER.dump_python(part, mode="json"),
            metadata=requests.metadata.get(part.tool_call_id) or {},
        )

    return [
        *(build(part, DeferredKind.APPROVAL) for part in requests.approvals),
        *(build(part, DeferredKind.CALL) for part in requests.calls),
    ]


def to_requests(paused: PausedRun) -> DeferredToolRequests:
    return DeferredToolRequests(
        approvals=[
            TOOL_CALL_PART_ADAPTER.validate_python(call.part)
            for call in paused.calls
            if call.kind is DeferredKind.APPROVAL
        ],
        calls=[
            TOOL_CALL_PART_ADAPTER.validate_python(call.part)
            for call in paused.calls
            if call.kind is DeferredKind.CALL
        ],
        metadata={
            call.tool_call_id: call.metadata for call in paused.calls if call.metadata
        },
    )


def to_results(paused: PausedRun) -> DeferredToolResults:
    """Results for a fully resolved paused run.

    Goes through the original requests' own `build_results` so a call id that
    doesn't belong to this pause raises here rather than silently resuming a
    run with a tool call left unanswered.
    """
    approvals: dict[str, bool | DeferredToolApprovalResult] = {}
    calls: dict[str, Any] = {}
    for call in paused.calls:
        resolution = call.resolution
        if resolution is None:
            continue
        if call.kind is DeferredKind.APPROVAL:
            approvals[call.tool_call_id] = (
                ToolApproved()
                if resolution.approved
                else ToolDenied(resolution.message or DEFAULT_DENIAL)
            )
        else:
            calls[call.tool_call_id] = resolution.content or MISSING_RESULT
    return to_requests(paused).build_results(
        approvals=approvals,
        calls=calls,
        metadata={
            call.tool_call_id: call.metadata for call in paused.calls if call.metadata
        },
    )


def unresolved_calls(paused: PausedRun) -> list[DeferredCall]:
    return [call for call in paused.calls if call.resolution is None]


async def record_pause(
    session_id: str,
    requests: DeferredToolRequests,
    intent: str,
    messages: Sequence[ModelMessage],
) -> PausedRun | None:
    calls = to_deferred_calls(requests)
    if not is_db_configured() or not calls:
        return None
    paused = await get_paused_run_store().replace(
        session_id=session_id,
        intent=intent,
        calls=calls,
        messages=serialize_messages(messages),
    )
    log_step(
        session_id,
        "park",
        f"stored {len(calls)} deferred call(s) and "
        f"{len(paused.messages)} paused message(s)",
    )
    return paused


async def find_paused(session_id: str) -> PausedRun | None:
    if not is_db_configured():
        return None
    return await get_paused_run_store().find(session_id)


def locate_call(
    paused: PausedRun, tool_call_id: str, kind: DeferredKind
) -> DeferredCall:
    for call in paused.calls:
        if call.tool_call_id != tool_call_id:
            continue
        if call.kind is not kind:
            raise UnknownDeferredCallError(
                f"{tool_call_id} is awaiting a {call.kind.value}, not a {kind.value}."
            )
        if call.resolution is not None:
            raise UnknownDeferredCallError(
                f"{tool_call_id} has already been answered."
            )
        return call
    raise UnknownDeferredCallError(
        f"{tool_call_id} is not awaiting a response for this session."
    )


async def require_paused(session_id: str) -> PausedRun:
    paused = await find_paused(session_id)
    if paused is None:
        raise UnknownDeferredCallError(
            "Nothing is awaiting a response for this session."
        )
    return paused


async def require_call(
    session_id: str, tool_call_id: str, kind: DeferredKind
) -> DeferredCall:
    """An unanswered deferred call, without resolving it.

    Separate from resolve_call because an external call's result has to be
    produced *from* what the call offered — the work happens between locating
    it and recording its outcome.
    """
    return locate_call(await require_paused(session_id), tool_call_id, kind)


async def resolve_call(
    session_id: str,
    tool_call_id: str,
    kind: DeferredKind,
    resolution: CallResolution,
) -> PausedRun:
    """Record one call's outcome on the paused run.

    Deliberately does not resume: a response can defer several calls, and the
    run is only replayable once every one of them has an answer.
    """
    paused = await require_paused(session_id)
    locate_call(paused, tool_call_id, kind).resolution = resolution
    log_step(session_id, "resolved", f"{kind.value} {tool_call_id}")
    return await get_paused_run_store().save(paused)


ABANDONED_DECLINE = (
    "The visitor moved on without responding, so this was not carried out."
)


async def decline_abandoned(session_id: str) -> bool:
    """Treat a still-pending pause as declined.

    Called when the visitor sends another message instead of answering: the
    offer is over, and leaving the record to expire would hide that the action
    was never taken. Every unanswered call in the pause is closed out together
    — a half-answered run cannot be replayed either.
    """
    paused = await find_paused(session_id)
    if paused is None:
        return False
    declined_turn = build_declined_turn(
        deserialize_turns([paused.messages]), ABANDONED_DECLINE
    )
    if declined_turn:
        await append_turn(session_id, declined_turn)
    await get_paused_run_store().discard(session_id)
    log_step(
        session_id,
        "abandon",
        f"closed out {len(unresolved_calls(paused))} unanswered call(s)",
    )
    return True


async def discard_paused(session_id: str) -> None:
    if not is_db_configured():
        return
    await get_paused_run_store().discard(session_id)
