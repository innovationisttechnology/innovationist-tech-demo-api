"""HTTP-level tests for the ziza_chat routes.

These pin the wire contract a client codes against: status codes, response
shapes, and which routes need a database. The unit test suite runs with
MONGO_URI unset (see conftest), so the database-gated routes answer 503 here
unless is_db_configured is patched.
"""

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.db import dependencies as db_dependencies
from app.main import app
from app.ziza_chat import service
from app.ziza_chat.hitl.models import DeferredKind
from app.ziza_chat.hitl.service import UnknownDeferredCallError
from app.ziza_chat.schemas import ChatResponse, PendingCallRead

# (method, request path, OpenAPI path template, request kwargs). The template
# is what the coverage check below compares against, so a new route cannot be
# added without a request that proves it is gated.
DB_GATED_REQUESTS = [
    ("post", "/api/ziza/chat", "/api/ziza/chat",
     {"json": {"session_id": "s", "message": "hi"}}),
    ("post", "/api/ziza/chat/stream", "/api/ziza/chat/stream",
     {"json": {"session_id": "s", "message": "hi"}}),
    ("post", "/api/ziza/chat/approval", "/api/ziza/chat/approval",
     {"json": {"session_id": "s", "tool_call_id": "c", "approved": True}}),
    ("post", "/api/ziza/chat/links", "/api/ziza/chat/links",
     {"json": {"session_id": "s", "tool_call_id": "c", "selected_links": []}}),
    ("post", "/api/ziza/knowledge/file", "/api/ziza/knowledge/file",
     {"data": {"session_id": "s"}, "files": {"file": ("a.txt", b"hi", "text/plain")}}),
    ("delete", "/api/ziza/knowledge/s", "/api/ziza/knowledge/{session_id}", {}),
]


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def allow_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db_dependencies, "is_db_configured", lambda: True)


class TestDatabaseGate:
    @pytest.mark.parametrize("method,path,template,payload", DB_GATED_REQUESTS)
    def test_a_gated_route_answers_503_without_a_database(
        self,
        client: TestClient,
        method: str,
        path: str,
        template: str,
        payload: dict[str, Any],
    ) -> None:
        response = getattr(client, method)(path, **payload)
        assert response.status_code == 503

    def test_every_route_is_gated(self) -> None:
        # Read from the OpenAPI schema, not app.routes: included routers are
        # nested behind a single opaque route object, so walking app.routes
        # finds no ziza paths at all and the check silently passes.
        covered = {template for _, _, template, _ in DB_GATED_REQUESTS}
        declared = {
            path for path in app.openapi()["paths"] if path.startswith("/api/ziza")
        }
        assert declared, "no ziza routes found — the check would be vacuous"
        assert declared - covered == set(), f"ungated ziza routes: {declared - covered}"


class TestChatResponseShape:
    def test_a_plain_answer_has_nothing_pending(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)

        async def fake_chat(request: Any) -> ChatResponse:
            return ChatResponse(
                session_id=request.session_id, response="an answer", intent="question"
            )

        monkeypatch.setattr(service, "chat", fake_chat)
        body = client.post(
            "/api/ziza/chat", json={"session_id": "s", "message": "hi"}
        ).json()
        assert body["pending_calls"] == []

    def test_a_paused_run_returns_the_call_id_to_confirm(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)

        async def fake_chat(request: Any) -> ChatResponse:
            return ChatResponse(
                session_id=request.session_id,
                response="Confirm to continue.",
                intent="task request",
                pending_calls=[
                    PendingCallRead(
                        tool_call_id="call_1",
                        tool_name="clear_knowledge_base",
                        kind=DeferredKind.APPROVAL,
                        details={"summary": "Delete 2 document(s)."},
                    )
                ],
            )

        monkeypatch.setattr(service, "chat", fake_chat)
        body = client.post(
            "/api/ziza/chat", json={"session_id": "s", "message": "clear it"}
        ).json()
        pending = body["pending_calls"][0]
        assert pending["tool_call_id"] == "call_1"
        assert pending["tool_name"] == "clear_knowledge_base"
        assert pending["kind"] == "approval"
        assert pending["details"]["summary"] == "Delete 2 document(s)."


class TestApprovalEndpoint:
    def test_a_decision_returns_the_resumed_answer(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)

        async def fake_resolve(request: Any) -> ChatResponse:
            assert request.approved is True
            assert request.tool_call_id == "call_1"
            return ChatResponse(
                session_id=request.session_id, response="Done.", intent="task request"
            )

        monkeypatch.setattr(service, "resolve_approval", fake_resolve)
        response = client.post(
            "/api/ziza/chat/approval",
            json={"session_id": "s", "tool_call_id": "call_1", "approved": True},
        )
        assert response.status_code == 200
        assert response.json()["response"] == "Done."

    def test_nothing_pending_is_a_conflict(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)

        async def fake_resolve(request: Any) -> ChatResponse:
            raise UnknownDeferredCallError("No approval is pending.")

        monkeypatch.setattr(service, "resolve_approval", fake_resolve)
        response = client.post(
            "/api/ziza/chat/approval",
            json={"session_id": "s", "tool_call_id": "stale", "approved": True},
        )
        assert response.status_code == 409
        assert "No approval is pending." in response.text

    def test_a_malformed_decision_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)
        response = client.post(
            "/api/ziza/chat/approval", json={"session_id": "s", "approved": True}
        )
        assert response.status_code == 422

    def test_the_database_gate_outranks_body_validation(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/ziza/chat/approval", json={"session_id": "s", "approved": True}
        )
        assert response.status_code == 503


class TestStreamFrames:
    def test_text_arrives_as_chat_chunk_frames(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.ziza_chat.schemas import ChatStreamEvent

        allow_db(monkeypatch)

        async def fake_stream(request: Any) -> Any:
            for chunk in ("Hello", " there"):
                yield ChatStreamEvent(type="chat.chunk", chunk=chunk)

        monkeypatch.setattr(service, "stream_chat", fake_stream)
        response = client.post(
            "/api/ziza/chat/stream", json={"session_id": "s", "message": "hi"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text.count("event: chat.chunk") == 2
        assert '"chunk": "Hello"' in response.text

    def test_a_pause_arrives_as_its_own_frame(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.ziza_chat.schemas import ChatStreamEvent

        allow_db(monkeypatch)

        async def fake_stream(request: Any) -> Any:
            yield ChatStreamEvent(type="chat.chunk", chunk="Confirm to continue.")
            yield ChatStreamEvent(
                type="chat.approval_required",
                pending_call=PendingCallRead(
                    tool_call_id="call_1",
                    tool_name="clear_knowledge_base",
                    kind=DeferredKind.APPROVAL,
                ),
            )

        monkeypatch.setattr(service, "stream_chat", fake_stream)
        response = client.post(
            "/api/ziza/chat/stream", json={"session_id": "s", "message": "clear it"}
        )
        assert "event: chat.approval_required" in response.text
        assert '"tool_call_id": "call_1"' in response.text


class TestLinkSelectionEndpoint:
    def test_a_selection_returns_the_resumed_answer(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)

        async def fake_resolve(request: Any) -> ChatResponse:
            assert request.selected_links == ["https://example.com/a"]
            return ChatResponse(
                session_id=request.session_id,
                response="Indexed 2 pages.",
                intent="task request",
            )

        monkeypatch.setattr(service, "resolve_link_selection", fake_resolve)
        response = client.post(
            "/api/ziza/chat/links",
            json={
                "session_id": "s",
                "tool_call_id": "call_1",
                "selected_links": ["https://example.com/a"],
            },
        )
        assert response.status_code == 200
        assert response.json()["response"] == "Indexed 2 pages."

    def test_an_empty_selection_is_valid(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Picking no links means "just the page" — not a malformed request."""
        allow_db(monkeypatch)

        async def fake_resolve(request: Any) -> ChatResponse:
            assert request.selected_links == []
            return ChatResponse(
                session_id=request.session_id, response="Indexed.", intent="task request"
            )

        monkeypatch.setattr(service, "resolve_link_selection", fake_resolve)
        response = client.post(
            "/api/ziza/chat/links",
            json={"session_id": "s", "tool_call_id": "call_1"},
        )
        assert response.status_code == 200

    def test_a_link_that_was_never_offered_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)

        async def fake_resolve(request: Any) -> ChatResponse:
            raise service.UnofferedLinkError("Not offered for this call: evil.com.")

        monkeypatch.setattr(service, "resolve_link_selection", fake_resolve)
        response = client.post(
            "/api/ziza/chat/links",
            json={
                "session_id": "s",
                "tool_call_id": "call_1",
                "selected_links": ["https://evil.com"],
            },
        )
        assert response.status_code == 422
        assert "Not offered" in response.text

    def test_nothing_pending_is_a_conflict(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_db(monkeypatch)

        async def fake_resolve(request: Any) -> ChatResponse:
            raise UnknownDeferredCallError("call_1 is not awaiting a response.")

        monkeypatch.setattr(service, "resolve_link_selection", fake_resolve)
        response = client.post(
            "/api/ziza/chat/links",
            json={"session_id": "s", "tool_call_id": "call_1"},
        )
        assert response.status_code == 409
