"""Tests for human-in-the-loop tool approval.

The gate must hold in code: the tool cannot delete anything until a decision
comes back, and a decision can only be spent once. Runs without MongoDB.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai import ApprovalRequired, DeferredToolRequests, RunContext
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.ziza_chat import service
from app.ziza_chat.agents.chat import get_chat_agent
from app.ziza_chat.agents.outputs import ClassifyResult, Intent, Scope
from app.ziza_chat.deps import ChatDeps
from app.ziza_chat.history_store import service as history_service
from app.ziza_chat.history_store.store import serialize_messages
from app.ziza_chat.hitl import service as hitl_service
from app.ziza_chat.schemas import (
    ApprovalDecisionRequest,
    ChatRequest,
    ChatStreamEvent,
)
from app.ziza_chat.tools import knowledge as knowledge_tool
from app.ziza_chat.utils import format_sse


class FakePending:
    """Stands in for the PendingApproval document.

    Constructing a Beanie Document calls get_settings(), which raises until
    init_beanie has run, and these tests deliberately need no database.
    """

    def __init__(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        intent: str,
        details: dict[str, Any],
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.session_id = session_id
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.intent = intent
        self.details = details
        self.messages = messages or []


class StubApprovalStore:
    def __init__(self, pending: FakePending | None = None) -> None:
        self.pending = pending
        self.discarded: list[str] = []

    async def replace(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        intent: str,
        details: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> FakePending:
        self.pending = FakePending(
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            intent=intent,
            details=details,
            messages=messages,
        )
        return self.pending

    async def find(self, session_id: str) -> FakePending | None:
        return self.pending

    async def discard(self, session_id: str) -> None:
        self.discarded.append(session_id)
        self.pending = None


def install_approval_store(
    monkeypatch: pytest.MonkeyPatch, pending: FakePending | None = None
) -> StubApprovalStore:
    store = StubApprovalStore(pending)
    monkeypatch.setattr(hitl_service, "get_approval_store", lambda: store)
    monkeypatch.setattr(hitl_service, "is_db_configured", lambda: True)
    return store


def tool_context(approved: bool) -> RunContext[ChatDeps]:
    return cast(
        RunContext[ChatDeps],
        SimpleNamespace(
            deps=ChatDeps(session_id="session-1"), tool_call_approved=approved
        ),
    )


def deferred_requests(tool_call_id: str = "call_1") -> DeferredToolRequests:
    return DeferredToolRequests(
        approvals=[
            ToolCallPart(
                tool_name="clear_knowledge_base",
                args={},
                tool_call_id=tool_call_id,
            )
        ],
        metadata={tool_call_id: {"summary": "Delete 2 document(s)."}},
    )


def pending_record(tool_call_id: str = "call_1") -> FakePending:
    return FakePending(
        session_id="session-1",
        tool_call_id=tool_call_id,
        tool_name="clear_knowledge_base",
        intent="task request",
        details={"summary": "Delete 2 document(s)."},
    )


class TestTheToolGate:
    @pytest.mark.anyio
    async def test_it_refuses_to_run_before_approval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cleared: list[str] = []

        async def holdings(session_id: str) -> list[str]:
            return ["handbook.pdf", "onboarding.md"]

        async def must_not_clear(session_id: str) -> int:
            cleared.append(session_id)
            return 0

        monkeypatch.setattr(knowledge_tool, "describe_holdings", holdings)
        monkeypatch.setattr(knowledge_tool, "clear_knowledge", must_not_clear)

        with pytest.raises(ApprovalRequired) as raised:
            await knowledge_tool.clear_knowledge_base(tool_context(approved=False))
        assert cleared == []
        assert raised.value.metadata is not None
        assert raised.value.metadata["documents"] == ["handbook.pdf", "onboarding.md"]

    @pytest.mark.anyio
    async def test_it_runs_once_approved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cleared: list[str] = []

        async def clear(session_id: str) -> int:
            cleared.append(session_id)
            return 7

        monkeypatch.setattr(knowledge_tool, "clear_knowledge", clear)
        result = await knowledge_tool.clear_knowledge_base(tool_context(approved=True))
        assert cleared == ["session-1"]
        assert "7" in result


class TestRecordingWhatIsPending:
    @pytest.mark.anyio
    async def test_a_paused_run_becomes_a_pending_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_approval_store(monkeypatch)
        pending = await hitl_service.record_pending(
            "session-1", deferred_requests(), "task request", []
        )
        assert pending is not None
        assert pending.tool_call_id == "call_1"
        assert store.pending is pending

    @pytest.mark.anyio
    async def test_the_response_carries_the_pending_approval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_approval_store(monkeypatch)
        response = await service.build_response(
            "session-1", "task request", deferred_requests()
        )
        assert response.pending_approval is not None
        assert response.pending_approval.tool_call_id == "call_1"
        assert "Delete 2 document(s)." in response.response

    @pytest.mark.anyio
    async def test_a_plain_answer_carries_no_approval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_approval_store(monkeypatch)
        response = await service.build_response("session-1", "question", "an answer")
        assert response.pending_approval is None
        assert response.response == "an answer"


class TestClaimingADecision:
    @pytest.mark.anyio
    async def test_a_decision_can_only_be_spent_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_approval_store(monkeypatch, pending_record())
        claimed = await hitl_service.claim_pending("session-1", "call_1")
        assert claimed.tool_call_id == "call_1"
        assert store.discarded == ["session-1"]
        with pytest.raises(hitl_service.NoPendingApprovalError):
            await hitl_service.claim_pending("session-1", "call_1")

    @pytest.mark.anyio
    async def test_a_mismatched_call_id_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_approval_store(monkeypatch, pending_record("call_1"))
        with pytest.raises(hitl_service.NoPendingApprovalError):
            await hitl_service.claim_pending("session-1", "someone_elses_call")

    @pytest.mark.anyio
    async def test_nothing_pending_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_approval_store(monkeypatch, None)
        with pytest.raises(hitl_service.NoPendingApprovalError):
            await hitl_service.claim_pending("session-1", "call_1")

    @pytest.mark.anyio
    async def test_resolving_without_a_pending_record_never_runs_the_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_approval_store(monkeypatch, None)

        def must_not_run() -> object:
            raise AssertionError("the agent ran without a pending approval")

        monkeypatch.setattr(service, "get_chat_agent", must_not_run)
        with pytest.raises(hitl_service.NoPendingApprovalError):
            await service.resolve_approval(
                ApprovalDecisionRequest(
                    session_id="session-1", tool_call_id="call_1", approved=True
                )
            )


class TestStreamEvents:
    def test_a_chunk_is_a_chat_chunk_frame(self) -> None:
        wire = format_sse(ChatStreamEvent(type="chat.chunk", chunk="hello"))
        assert wire.startswith("event: chat.chunk\n")
        assert '"chunk": "hello"' in wire

    def test_an_approval_frame_carries_the_call_id(self) -> None:
        event = ChatStreamEvent(
            type="chat.approval_required",
            pending_approval=service.to_pending_read(cast(Any, pending_record())),
        )
        wire = format_sse(event)
        assert wire.startswith("event: chat.approval_required\n")
        assert '"tool_call_id": "call_1"' in wire

    def test_empty_fields_are_left_out_of_the_frame(self) -> None:
        wire = format_sse(ChatStreamEvent(type="chat.chunk", chunk="hi"))
        assert "pending_approval" not in wire


class StubVectorStore:
    async def list_documents(self, session_id: str) -> list[str]:
        return ["handbook.pdf", "onboarding.md"]


def install_paused_stream(monkeypatch: pytest.MonkeyPatch) -> StubApprovalStore:
    """Wire a session where the model reaches straight for the gated tool."""

    async def fake_classify(
        message: str, previous_message: str | None = None
    ) -> ClassifyResult:
        return ClassifyResult(
            scope=Scope.ASSISTANT,
            intents=[Intent.TASK_REQUEST],
            needs_rag=False,
            rag_query=None,
        )

    async def holdings(session_id: str) -> list[str]:
        return ["handbook.pdf", "onboarding.md"]

    async def no_history(session_id: str) -> list[Any]:
        return []

    async def swallow_turn(session_id: str, messages: Any) -> None:
        return None

    monkeypatch.setattr(service, "classify", fake_classify)
    monkeypatch.setattr(service, "get_vector_store", lambda: StubVectorStore())
    monkeypatch.setattr(service, "is_db_configured", lambda: True)
    monkeypatch.setattr(service, "load_history", no_history)
    monkeypatch.setattr(service, "append_turn", swallow_turn)
    monkeypatch.setattr(history_service, "is_db_configured", lambda: False)
    monkeypatch.setattr(knowledge_tool, "describe_holdings", holdings)
    return install_approval_store(monkeypatch)


class TestStreamingWithTheGatedTool:
    @pytest.mark.anyio
    async def test_a_pause_does_not_blow_up_the_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stream_text() raises on a non-text output; stream_output() does not."""
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(
            model=TestModel(call_tools=["clear_knowledge_base"])
        ):
            events = [
                event
                async for event in service.stream_chat(
                    ChatRequest(session_id="session-1", message="clear my documents")
                )
            ]
        assert events

    @pytest.mark.anyio
    async def test_the_stream_ends_with_an_approval_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(
            model=TestModel(call_tools=["clear_knowledge_base"])
        ):
            events = [
                event
                async for event in service.stream_chat(
                    ChatRequest(session_id="session-1", message="clear my documents")
                )
            ]
        assert events[-1].type == "chat.approval_required"
        pending = events[-1].pending_approval
        assert pending is not None
        assert pending.tool_name == "clear_knowledge_base"

    @pytest.mark.anyio
    async def test_the_visitor_is_told_what_needs_confirming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(
            model=TestModel(call_tools=["clear_knowledge_base"])
        ):
            events = [
                event
                async for event in service.stream_chat(
                    ChatRequest(session_id="session-1", message="clear my documents")
                )
            ]
        chunks = " ".join(event.chunk or "" for event in events)
        assert "Delete 2 document(s)" in chunks

    @pytest.mark.anyio
    async def test_a_plain_answer_still_streams_as_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(model=TestModel(call_tools=[])):
            events = [
                event
                async for event in service.stream_chat(
                    ChatRequest(session_id="session-1", message="what can you do?")
                )
            ]
        assert [event.type for event in events] == ["chat.chunk"] * len(events)
        assert "".join(event.chunk or "" for event in events)


def paused_messages() -> list[Any]:
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

    return [
        ModelRequest(parts=[UserPromptPart(content="clear my documents")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="clear_knowledge_base", args={}, tool_call_id="call_1"
                )
            ]
        ),
    ]


class TestAbandonedApprovals:
    @pytest.mark.anyio
    async def test_moving_on_counts_as_declining(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = pending_record()
        record.messages = serialize_messages(paused_messages())
        store = install_approval_store(monkeypatch, record)
        appended: list[list[Any]] = []

        async def capture(session_id: str, messages: Any) -> None:
            appended.append(list(messages))

        monkeypatch.setattr(hitl_service, "append_turn", capture)
        assert await hitl_service.decline_abandoned("session-1") is True
        assert store.discarded == ["session-1"]
        assert appended, "the declined turn was not recorded"

    @pytest.mark.anyio
    async def test_the_declined_turn_answers_the_dangling_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unanswered tool call is what bricked a live session with a 400."""
        record = pending_record()
        record.messages = serialize_messages(paused_messages())
        install_approval_store(monkeypatch, record)
        appended: list[list[Any]] = []

        async def capture(session_id: str, messages: Any) -> None:
            appended.append(list(messages))

        monkeypatch.setattr(hitl_service, "append_turn", capture)
        await hitl_service.decline_abandoned("session-1")
        parts = [part for message in appended[0] for part in message.parts]
        offered = {p.tool_call_id for p in parts if isinstance(p, ToolCallPart)}
        answered = {p.tool_call_id for p in parts if isinstance(p, ToolReturnPart)}
        assert offered == answered == {"call_1"}

    @pytest.mark.anyio
    async def test_nothing_pending_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_approval_store(monkeypatch, None)
        assert await hitl_service.decline_abandoned("session-1") is False
        assert store.discarded == []
