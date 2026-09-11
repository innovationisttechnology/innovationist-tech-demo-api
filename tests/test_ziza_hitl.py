"""Tests for deferred tool calls: approvals and externally executed calls.

The gate must hold in code. A gated tool cannot act until a decision comes
back, a decision can only be spent once, and a run holding several deferred
calls must not be replayed until every one of them has an answer. Runs without
MongoDB.
"""

from types import SimpleNamespace
from typing import Any, Sequence, cast

import pytest
from pydantic_ai import (
    ApprovalRequired,
    CallDeferred,
    DeferredToolRequests,
    RunContext,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel

from app.ziza_chat import service
from app.ziza_chat.agents.chat import get_chat_agent
from app.ziza_chat.agents.outputs import ClassifyResult, Intent, Scope
from app.ziza_chat.deps import ChatDeps
from app.ziza_chat.document_loaders.base import UnsupportedDocumentError
from app.ziza_chat.document_loaders.web import (
    CandidateLink,
    FetchedPage,
    UnsafeUrlError,
)
from app.ziza_chat.history_store import service as history_service
from app.ziza_chat.history_store.store import serialize_messages
from app.ziza_chat.hitl import service as hitl_service
from app.ziza_chat.hitl.models import CallResolution, DeferredCall, DeferredKind
from app.ziza_chat.schemas import (
    ApprovalDecisionRequest,
    ChatRequest,
    ChatStreamEvent,
    LinkSelectionRequest,
)
from app.ziza_chat.tools import knowledge as knowledge_tool
from app.ziza_chat.utils import format_sse


class FakePausedRun:
    """Stands in for the PausedRun document.

    Constructing a Beanie Document calls get_settings(), which raises until
    init_beanie has run, and these tests deliberately need no database.
    """

    def __init__(
        self,
        session_id: str = "session-1",
        intent: str = "task request",
        calls: Sequence[DeferredCall] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.session_id = session_id
        self.intent = intent
        self.calls = list(calls or [])
        self.messages = messages or []


class StubPausedRunStore:
    def __init__(self, paused: FakePausedRun | None = None) -> None:
        self.paused = paused
        self.discarded: list[str] = []

    async def replace(
        self,
        session_id: str,
        intent: str,
        calls: Sequence[DeferredCall],
        messages: list[dict[str, Any]],
    ) -> FakePausedRun:
        self.paused = FakePausedRun(session_id, intent, calls, messages)
        return self.paused

    async def find(self, session_id: str) -> FakePausedRun | None:
        return self.paused

    async def save(self, paused: FakePausedRun) -> FakePausedRun:
        return paused

    async def discard(self, session_id: str) -> None:
        self.discarded.append(session_id)
        self.paused = None


def install_store(
    monkeypatch: pytest.MonkeyPatch, paused: FakePausedRun | None = None
) -> StubPausedRunStore:
    store = StubPausedRunStore(paused)
    monkeypatch.setattr(hitl_service, "get_paused_run_store", lambda: store)
    monkeypatch.setattr(hitl_service, "is_db_configured", lambda: True)
    return store


def tool_context(
    approved: bool = False, documents: Sequence[str] = ()
) -> RunContext[ChatDeps]:
    return cast(
        RunContext[ChatDeps],
        SimpleNamespace(
            deps=ChatDeps(session_id="session-1", documents=list(documents)),
            tool_call_approved=approved,
        ),
    )


def approval_part(tool_call_id: str = "call_1") -> ToolCallPart:
    return ToolCallPart(
        tool_name="clear_knowledge_base", args={}, tool_call_id=tool_call_id
    )


def link_part(tool_call_id: str = "call_2") -> ToolCallPart:
    return ToolCallPart(
        tool_name="add_url_to_knowledge_base",
        args={"url": "https://example.com/guide"},
        tool_call_id=tool_call_id,
    )


def deferred_requests(
    approvals: Sequence[ToolCallPart] = (),
    calls: Sequence[ToolCallPart] = (),
    metadata: dict[str, dict[str, Any]] | None = None,
) -> DeferredToolRequests:
    return DeferredToolRequests(
        approvals=list(approvals),
        calls=list(calls),
        metadata=metadata or {},
    )


def approval_only() -> DeferredToolRequests:
    return deferred_requests(
        approvals=[approval_part()],
        metadata={"call_1": {"summary": "Delete 2 document(s)."}},
    )


def link_offer(tool_call_id: str = "call_2") -> DeferredCall:
    return DeferredCall(
        tool_call_id=tool_call_id,
        tool_name="add_url_to_knowledge_base",
        kind=DeferredKind.CALL,
        part=hitl_service.TOOL_CALL_PART_ADAPTER.dump_python(
            link_part(tool_call_id), mode="json"
        ),
        metadata={
            "action": "add_url_to_knowledge_base",
            "url": "https://example.com/guide",
            "links": [
                {"url": "https://example.com/guide/setup", "text": "Setup"},
                {"url": "https://example.com/guide/api", "text": "API"},
            ],
            "summary": "Index the guide.",
        },
    )


class TestTheApprovalGate:
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


def install_fetch(
    monkeypatch: pytest.MonkeyPatch, links: Sequence[CandidateLink] = ()
) -> list[str]:
    fetched: list[str] = []

    async def fetch(url: str) -> FetchedPage:
        fetched.append(url)
        return FetchedPage(
            url=url, title="The Guide", content_type="text/html", body=b"<html></html>"
        )

    monkeypatch.setattr(knowledge_tool, "fetch_page", fetch)
    monkeypatch.setattr(
        knowledge_tool, "extract_links", lambda body, base_url: list(links)
    )
    return fetched


class TestTheUrlTool:
    @pytest.mark.anyio
    async def test_it_offers_the_links_instead_of_indexing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tool body never runs again, so it must not index anything."""
        install_fetch(
            monkeypatch,
            [CandidateLink(url="https://example.com/a", text="A")],
        )
        with pytest.raises(CallDeferred) as raised:
            await knowledge_tool.add_url_to_knowledge_base(
                tool_context(), "https://example.com/guide"
            )
        metadata = raised.value.metadata
        assert metadata is not None
        assert metadata["url"] == "https://example.com/guide"
        assert metadata["links"] == [{"url": "https://example.com/a", "text": "A"}]
        assert metadata["slots_left"] == 10

    @pytest.mark.anyio
    async def test_a_page_with_no_links_still_defers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Indexing costs a session slot, so it is still the visitor's call."""
        install_fetch(monkeypatch)
        with pytest.raises(CallDeferred) as raised:
            await knowledge_tool.add_url_to_knowledge_base(
                tool_context(), "https://example.com/guide"
            )
        assert raised.value.metadata is not None
        assert raised.value.metadata["links"] == []

    @pytest.mark.anyio
    async def test_a_full_session_is_answered_not_deferred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetched = install_fetch(monkeypatch)
        result = await knowledge_tool.add_url_to_knowledge_base(
            tool_context(documents=[f"doc-{index}" for index in range(10)]),
            "https://example.com/guide",
        )
        assert "limit" in result
        assert fetched == [], "a full session should not cost a fetch"

    @pytest.mark.anyio
    async def test_an_unreachable_page_is_answered_not_deferred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def refuse(url: str) -> FetchedPage:
            raise UnsafeUrlError("resolves to a non-public address")

        monkeypatch.setattr(knowledge_tool, "fetch_page", refuse)
        result = await knowledge_tool.add_url_to_knowledge_base(
            tool_context(), "http://169.254.169.254/"
        )
        assert "Could not read" in result


class TestRecordingAPause:
    @pytest.mark.anyio
    async def test_an_approval_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch)
        paused = await hitl_service.record_pause(
            "session-1", approval_only(), "task request", []
        )
        assert paused is not None
        assert [call.kind for call in paused.calls] == [DeferredKind.APPROVAL]
        assert paused.calls[0].metadata["summary"] == "Delete 2 document(s)."

    @pytest.mark.anyio
    async def test_an_external_call_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The old shape only looked at `approvals` and dropped these silently."""
        install_store(monkeypatch)
        paused = await hitl_service.record_pause(
            "session-1",
            deferred_requests(calls=[link_part()], metadata={"call_2": {"url": "u"}}),
            "task request",
            [],
        )
        assert paused is not None
        assert [call.kind for call in paused.calls] == [DeferredKind.CALL]
        assert paused.calls[0].tool_name == "add_url_to_knowledge_base"

    @pytest.mark.anyio
    async def test_every_call_in_one_response_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch)
        paused = await hitl_service.record_pause(
            "session-1",
            deferred_requests(
                approvals=[approval_part("call_1")],
                calls=[link_part("call_2"), link_part("call_3")],
            ),
            "task request",
            [],
        )
        assert paused is not None
        assert {call.tool_call_id for call in paused.calls} == {
            "call_1",
            "call_2",
            "call_3",
        }

    @pytest.mark.anyio
    async def test_nothing_deferred_records_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch)
        assert (
            await hitl_service.record_pause(
                "session-1", deferred_requests(), "question", []
            )
            is None
        )

    @pytest.mark.anyio
    async def test_the_tool_call_part_round_trips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stored part is what rebuilds the requests the results validate against."""
        install_store(monkeypatch)
        paused = await hitl_service.record_pause(
            "session-1", deferred_requests(calls=[link_part()]), "task request", []
        )
        assert paused is not None
        rebuilt = hitl_service.to_requests(cast(Any, paused))
        assert rebuilt.calls[0].args == {"url": "https://example.com/guide"}
        assert rebuilt.calls[0].tool_call_id == "call_2"


class TestBuildingTheResults:
    def test_an_approval_becomes_tool_approved(self) -> None:
        paused = FakePausedRun(
            calls=[
                DeferredCall(
                    tool_call_id="call_1",
                    tool_name="clear_knowledge_base",
                    kind=DeferredKind.APPROVAL,
                    part=hitl_service.TOOL_CALL_PART_ADAPTER.dump_python(
                        approval_part(), mode="json"
                    ),
                    resolution=CallResolution(approved=True),
                )
            ]
        )
        results = hitl_service.to_results(cast(Any, paused))
        assert isinstance(results.approvals["call_1"], ToolApproved)

    def test_a_refusal_becomes_tool_denied(self) -> None:
        paused = FakePausedRun(
            calls=[
                DeferredCall(
                    tool_call_id="call_1",
                    tool_name="clear_knowledge_base",
                    kind=DeferredKind.APPROVAL,
                    part=hitl_service.TOOL_CALL_PART_ADAPTER.dump_python(
                        approval_part(), mode="json"
                    ),
                    resolution=CallResolution(approved=False),
                )
            ]
        )
        results = hitl_service.to_results(cast(Any, paused))
        denial = results.approvals["call_1"]
        assert isinstance(denial, ToolDenied)
        assert denial.message == hitl_service.DEFAULT_DENIAL

    def test_an_external_result_is_the_tool_return(self) -> None:
        offer = link_offer()
        offer.resolution = CallResolution(content="Indexed 3 pages.")
        paused = FakePausedRun(calls=[offer])
        results = hitl_service.to_results(cast(Any, paused))
        assert results.calls["call_2"] == "Indexed 3 pages."

    def test_the_metadata_is_carried_into_the_resumed_run(self) -> None:
        offer = link_offer()
        offer.resolution = CallResolution(content="Indexed.")
        paused = FakePausedRun(calls=[offer])
        results = hitl_service.to_results(cast(Any, paused))
        assert results.metadata["call_2"]["url"] == "https://example.com/guide"


class TestResolvingOneCall:
    @pytest.mark.anyio
    async def test_a_decision_can_only_be_spent_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(
            monkeypatch,
            FakePausedRun(
                calls=[
                    DeferredCall(
                        tool_call_id="call_1",
                        tool_name="clear_knowledge_base",
                        kind=DeferredKind.APPROVAL,
                    )
                ]
            ),
        )
        await hitl_service.resolve_call(
            "session-1", "call_1", DeferredKind.APPROVAL, CallResolution(approved=True)
        )
        with pytest.raises(hitl_service.UnknownDeferredCallError):
            await hitl_service.resolve_call(
                "session-1",
                "call_1",
                DeferredKind.APPROVAL,
                CallResolution(approved=True),
            )

    @pytest.mark.anyio
    async def test_the_wrong_kind_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An external call's result is not a yes/no, and vice versa."""
        install_store(monkeypatch, FakePausedRun(calls=[link_offer()]))
        with pytest.raises(hitl_service.UnknownDeferredCallError):
            await hitl_service.resolve_call(
                "session-1",
                "call_2",
                DeferredKind.APPROVAL,
                CallResolution(approved=True),
            )

    @pytest.mark.anyio
    async def test_a_mismatched_call_id_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, FakePausedRun(calls=[link_offer()]))
        with pytest.raises(hitl_service.UnknownDeferredCallError):
            await hitl_service.require_call(
                "session-1", "someone_elses_call", DeferredKind.CALL
            )

    @pytest.mark.anyio
    async def test_nothing_paused_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, None)
        with pytest.raises(hitl_service.UnknownDeferredCallError):
            await hitl_service.require_call("session-1", "call_1", DeferredKind.CALL)

    @pytest.mark.anyio
    async def test_resolving_without_a_pause_never_runs_the_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, None)

        def must_not_run() -> object:
            raise AssertionError("the agent ran with nothing paused")

        monkeypatch.setattr(service, "get_chat_agent", must_not_run)
        with pytest.raises(hitl_service.UnknownDeferredCallError):
            await service.resolve_approval(
                ApprovalDecisionRequest(
                    session_id="session-1", tool_call_id="call_1", approved=True
                )
            )


class TestWaitingOnTheRest:
    @pytest.mark.anyio
    async def test_a_half_answered_run_is_not_replayed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tool call with no result is rejected on replay, so it must wait."""
        install_store(
            monkeypatch,
            FakePausedRun(
                calls=[
                    DeferredCall(
                        tool_call_id="call_1",
                        tool_name="clear_knowledge_base",
                        kind=DeferredKind.APPROVAL,
                    ),
                    link_offer("call_2"),
                ]
            ),
        )

        def must_not_run() -> object:
            raise AssertionError("the agent ran with a call still unanswered")

        monkeypatch.setattr(service, "get_chat_agent", must_not_run)
        response = await service.resolve_approval(
            ApprovalDecisionRequest(
                session_id="session-1", tool_call_id="call_1", approved=True
            )
        )
        assert [pending.tool_call_id for pending in response.pending_calls] == ["call_2"]


class TestStreamEvents:
    def test_a_chunk_is_a_chat_chunk_frame(self) -> None:
        wire = format_sse(ChatStreamEvent(type="chat.chunk", chunk="hello"))
        assert wire.startswith("event: chat.chunk\n")
        assert '"chunk": "hello"' in wire

    def test_an_approval_frame_carries_the_call_id(self) -> None:
        event = ChatStreamEvent(
            type="chat.approval_required",
            pending_call=service.to_pending_read(
                DeferredCall(
                    tool_call_id="call_1",
                    tool_name="clear_knowledge_base",
                    kind=DeferredKind.APPROVAL,
                )
            ),
        )
        wire = format_sse(event)
        assert wire.startswith("event: chat.approval_required\n")
        assert '"tool_call_id": "call_1"' in wire

    def test_empty_fields_are_left_out_of_the_frame(self) -> None:
        wire = format_sse(ChatStreamEvent(type="chat.chunk", chunk="hi"))
        assert "pending_call" not in wire


class StubVectorStore:
    async def list_documents(self, session_id: str) -> list[str]:
        return ["handbook.pdf", "onboarding.md"]


def install_paused_stream(monkeypatch: pytest.MonkeyPatch) -> StubPausedRunStore:
    """Wire a session where the model reaches straight for a gated tool."""

    async def fake_classify(
        message: str,
        previous_message: str | None = None,
        session_id: str = "-",
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
    return install_store(monkeypatch)


async def stream_events(message: str) -> list[ChatStreamEvent]:
    return [
        event
        async for event in service.stream_chat(
            ChatRequest(session_id="session-1", message=message)
        )
    ]


class TestStreamingWithADeferredTool:
    @pytest.mark.anyio
    async def test_a_pause_does_not_blow_up_the_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stream_text() raises on a non-text output; stream_output() does not."""
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(
            model=TestModel(call_tools=["clear_knowledge_base"])
        ):
            assert await stream_events("clear my documents")

    @pytest.mark.anyio
    async def test_an_approval_ends_with_an_approval_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(
            model=TestModel(call_tools=["clear_knowledge_base"])
        ):
            events = await stream_events("clear my documents")
        assert events[-1].type == "chat.approval_required"
        pending = events[-1].pending_call
        assert pending is not None
        assert pending.tool_name == "clear_knowledge_base"
        assert pending.kind is DeferredKind.APPROVAL

    @pytest.mark.anyio
    async def test_an_external_call_ends_with_an_input_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_paused_stream(monkeypatch)
        install_fetch(
            monkeypatch, [CandidateLink(url="https://example.com/a", text="A")]
        )
        with get_chat_agent().override(
            model=TestModel(call_tools=["add_url_to_knowledge_base"])
        ):
            events = await stream_events("add https://example.com/guide")
        assert events[-1].type == "chat.input_required"
        pending = events[-1].pending_call
        assert pending is not None
        assert pending.kind is DeferredKind.CALL
        assert pending.details["links"] == [
            {"url": "https://example.com/a", "text": "A"}
        ]

    @pytest.mark.anyio
    async def test_the_visitor_is_told_what_needs_confirming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(
            model=TestModel(call_tools=["clear_knowledge_base"])
        ):
            events = await stream_events("clear my documents")
        chunks = " ".join(event.chunk or "" for event in events)
        assert "Delete 2 document(s)" in chunks

    @pytest.mark.anyio
    async def test_a_plain_answer_still_streams_as_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_paused_stream(monkeypatch)
        with get_chat_agent().override(model=TestModel(call_tools=[])):
            events = await stream_events("what can you do?")
        assert [event.type for event in events] == ["chat.chunk"] * len(events)
        assert "".join(event.chunk or "" for event in events)


def paused_messages(*tool_call_ids: str) -> list[Any]:
    return [
        ModelRequest(parts=[UserPromptPart(content="clear my documents")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="clear_knowledge_base",
                    args={},
                    tool_call_id=tool_call_id,
                )
                for tool_call_id in tool_call_ids
            ]
        ),
    ]


class TestAbandonedPauses:
    @pytest.mark.anyio
    async def test_moving_on_counts_as_declining(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_store(
            monkeypatch,
            FakePausedRun(messages=serialize_messages(paused_messages("call_1"))),
        )
        appended: list[list[Any]] = []

        async def capture(session_id: str, messages: Any) -> None:
            appended.append(list(messages))

        monkeypatch.setattr(hitl_service, "append_turn", capture)
        assert await hitl_service.decline_abandoned("session-1") is True
        assert store.discarded == ["session-1"]
        assert appended, "the declined turn was not recorded"

    @pytest.mark.anyio
    async def test_the_declined_turn_answers_every_dangling_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unanswered tool call is what bricked a live session with a 400."""
        install_store(
            monkeypatch,
            FakePausedRun(
                messages=serialize_messages(paused_messages("call_1", "call_2"))
            ),
        )
        appended: list[list[Any]] = []

        async def capture(session_id: str, messages: Any) -> None:
            appended.append(list(messages))

        monkeypatch.setattr(hitl_service, "append_turn", capture)
        await hitl_service.decline_abandoned("session-1")
        parts = [part for message in appended[0] for part in message.parts]
        offered = {p.tool_call_id for p in parts if isinstance(p, ToolCallPart)}
        answered = {p.tool_call_id for p in parts if isinstance(p, ToolReturnPart)}
        assert offered == answered == {"call_1", "call_2"}

    @pytest.mark.anyio
    async def test_nothing_paused_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_store(monkeypatch, None)
        assert await hitl_service.decline_abandoned("session-1") is False
        assert store.discarded == []


class TestExecutingTheLinkSelection:
    @pytest.mark.anyio
    async def test_a_link_that_was_never_offered_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The selection is what this endpoint goes and fetches."""
        install_store(monkeypatch, FakePausedRun(calls=[link_offer()]))
        indexed: list[str] = []
        monkeypatch.setattr(
            service,
            "ingest_url",
            lambda session, url, offer_links=True: indexed.append(url),
        )
        with pytest.raises(service.UnofferedLinkError):
            await service.resolve_link_selection(
                LinkSelectionRequest(
                    session_id="session-1",
                    tool_call_id="call_2",
                    selected_links=["https://evil.example.net/steal"],
                )
            )
        assert indexed == []

    @pytest.mark.anyio
    async def test_the_page_and_the_selection_are_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, FakePausedRun(calls=[link_offer()]))
        indexed: list[str] = []

        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            indexed.append(url)
            return SimpleNamespace(source=url, chunks_ingested=4)

        monkeypatch.setattr(service, "ingest_url", fake_ingest)
        outcome = await service.ingest_link_selection(
            "session-1",
            "https://example.com/guide",
            ["https://example.com/guide/setup"],
        )
        assert indexed == [
            "https://example.com/guide",
            "https://example.com/guide/setup",
        ]
        assert "4 passages" in outcome

    @pytest.mark.anyio
    async def test_one_unreachable_link_does_not_lose_the_rest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            if url.endswith("/setup"):
                raise UnsupportedDocumentError("no readable text")
            return SimpleNamespace(source=url, chunks_ingested=2)

        monkeypatch.setattr(service, "ingest_url", fake_ingest)
        outcome = await service.ingest_link_selection(
            "session-1",
            "https://example.com/guide",
            ["https://example.com/guide/setup"],
        )
        assert "Indexed and now searchable" in outcome
        assert "Could not index" in outcome
        assert "/setup" in outcome

    @pytest.mark.anyio
    async def test_the_result_resumes_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        offer = link_offer()
        store = install_store(
            monkeypatch,
            FakePausedRun(
                calls=[offer], messages=serialize_messages(paused_messages("call_2"))
            ),
        )
        supplied: list[Any] = []

        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            return SimpleNamespace(source=url, chunks_ingested=1)

        async def no_history(session_id: str) -> list[Any]:
            return []

        async def swallow_turn(session_id: str, messages: Any) -> None:
            return None

        async def fake_run(**kwargs: Any) -> Any:
            supplied.append(kwargs["deferred_tool_results"])
            return SimpleNamespace(
                output="Indexed the guide.", new_messages=lambda: []
            )

        monkeypatch.setattr(service, "ingest_url", fake_ingest)
        monkeypatch.setattr(service, "load_history", no_history)
        monkeypatch.setattr(service, "append_turn", swallow_turn)
        monkeypatch.setattr(service, "is_db_configured", lambda: False)
        monkeypatch.setattr(
            service, "get_chat_agent", lambda: SimpleNamespace(run=fake_run)
        )

        response = await service.resolve_link_selection(
            LinkSelectionRequest(
                session_id="session-1",
                tool_call_id="call_2",
                selected_links=["https://example.com/guide/api"],
            )
        )
        assert response.response == "Indexed the guide."
        assert response.pending_calls == []
        assert store.discarded == ["session-1"]
        assert "Indexed and now searchable" in supplied[0].calls["call_2"]


class TestTheWholeDeferredRoundTrip:
    """One pass through the real agent: defer, execute outside, resume.

    Hermetic — FunctionModel stands in for the provider — but every other
    piece is the production path, including pydantic-ai's own resume.
    """

    @pytest.mark.anyio
    async def test_the_tool_body_is_not_re_entered_on_resume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_paused_stream(monkeypatch)
        fetched = install_fetch(
            monkeypatch, [CandidateLink(url="https://example.com/a", text="A")]
        )
        indexed: list[str] = []

        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            indexed.append(url)
            return SimpleNamespace(source=url, chunks_ingested=3)

        monkeypatch.setattr(service, "ingest_url", fake_ingest)

        def already_answered(messages: list[ModelMessage]) -> bool:
            return any(
                isinstance(part, ToolReturnPart)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )

        def responder(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if already_answered(messages):
                return ModelResponse(parts=[TextPart(content="Added the guide.")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="add_url_to_knowledge_base",
                        args={"url": "https://example.com/guide"},
                        tool_call_id="call_1",
                    )
                ]
            )

        async def stream_responder(
            messages: list[ModelMessage], info: AgentInfo
        ) -> Any:
            # event_stream_handler makes run() stream internally, so it is
            # the streaming half of FunctionModel that gets used.
            if already_answered(messages):
                yield "Added the guide."
                return
            yield cast(
                DeltaToolCalls,
                {
                    0: DeltaToolCall(
                        name="add_url_to_knowledge_base",
                        json_args='{"url": "https://example.com/guide"}',
                        tool_call_id="call_1",
                    )
                },
            )

        model = FunctionModel(responder, stream_function=stream_responder)
        with get_chat_agent().override(model=model):
            paused_response = await service.chat(
                ChatRequest(
                    session_id="session-1",
                    message="add https://example.com/guide",
                )
            )
            assert [pending.kind for pending in paused_response.pending_calls] == [
                DeferredKind.CALL
            ]
            assert fetched == ["https://example.com/guide"]
            assert indexed == [], "nothing may be indexed before the visitor answers"

            resumed = await service.resolve_link_selection(
                LinkSelectionRequest(
                    session_id="session-1",
                    tool_call_id="call_1",
                    selected_links=["https://example.com/a"],
                )
            )

        assert resumed.response == "Added the guide."
        assert resumed.pending_calls == []
        assert indexed == ["https://example.com/guide", "https://example.com/a"]
        # A second fetch would mean the deferred body ran again.
        assert fetched == ["https://example.com/guide"]
        assert store.discarded == ["session-1"]


class TestOfferingAPageAlreadyHeld:
    @pytest.mark.anyio
    async def test_links_already_indexed_are_not_offered_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fetch(
            monkeypatch,
            [
                CandidateLink(url="https://example.com/held", text="Held"),
                CandidateLink(url="https://example.com/new", text="New"),
            ],
        )
        with pytest.raises(CallDeferred) as raised:
            await knowledge_tool.add_url_to_knowledge_base(
                tool_context(
                    documents=[
                        "https://example.com/guide",
                        "https://example.com/held",
                    ]
                ),
                "https://example.com/guide",
            )
        metadata = raised.value.metadata
        assert metadata is not None
        assert metadata["links"] == [{"url": "https://example.com/new", "text": "New"}]
        assert metadata["page_already_indexed"] is True
        assert "already indexed" in metadata["summary"]

    @pytest.mark.anyio
    async def test_an_already_held_page_is_not_re_ingested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexed: list[str] = []

        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            indexed.append(url)
            return SimpleNamespace(source=url, chunks_ingested=3)

        monkeypatch.setattr(service, "ingest_url", fake_ingest)
        await service.ingest_link_selection(
            "session-1",
            "https://example.com/guide",
            ["https://example.com/new"],
            index_base_url=False,
        )
        assert indexed == ["https://example.com/new"]

    @pytest.mark.anyio
    async def test_choosing_nothing_from_a_held_page_indexes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def must_not_run(session_id: str, url: str, **kwargs: Any) -> Any:
            raise AssertionError("nothing was selected")

        monkeypatch.setattr(service, "ingest_url", must_not_run)
        outcome = await service.ingest_link_selection(
            "session-1", "https://example.com/guide", [], index_base_url=False
        )
        assert "chose not to add" in outcome
