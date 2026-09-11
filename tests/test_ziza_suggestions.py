"""Tests for the pages offered alongside an answer.

The offer is not a deferred call: the run completed, the answer stands, and
ignoring it costs nothing. It fires only where retrieval already failed, which
is what keeps it from nagging.
"""

from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from app.ziza_chat import service
from app.ziza_chat.deps import ChatDeps, SearchOutcome
from app.ziza_chat.hitl.models import DeferredKind
from app.ziza_chat.page_links import StoredLink
from app.ziza_chat.schemas import (
    ChatRequest,
    ChatResponse,
    PendingCallRead,
    SuggestionKind,
)
from app.ziza_chat.vector_store.store import RetrievedChunk


def links(*urls: str) -> list[Any]:
    return [
        SimpleNamespace(
            document="https://example.com/",
            links=[StoredLink(url=url, text=url.rsplit("/", 1)[-1]) for url in urls],
        )
    ]


def install_links(
    monkeypatch: pytest.MonkeyPatch, records: list[Any] | None = None
) -> None:
    async def load(session_id: str) -> list[Any]:
        return records if records is not None else links("https://example.com/a")

    monkeypatch.setattr(service, "load_page_links", load)
    monkeypatch.setattr(service, "is_db_configured", lambda: True)


class TestReadingHowRetrievalWent:
    def test_a_run_that_never_searched_is_not_weak(self) -> None:
        """The agent answering without retrieval is not a retrieval failure."""
        assert service.retrieval_was_weak(ChatDeps(session_id="s")) is False

    def test_a_search_that_matched_nothing_is_weak(self) -> None:
        deps = ChatDeps(session_id="s")
        deps.searches.append(SearchOutcome(query="file", matches=0, best_score=None))
        assert service.retrieval_was_weak(deps) is True

    def test_one_good_search_is_enough_to_not_be_weak(self) -> None:
        deps = ChatDeps(session_id="s")
        deps.searches.append(SearchOutcome(query="file", matches=0, best_score=None))
        deps.searches.append(SearchOutcome(query="pricing", matches=2, best_score=0.8))
        assert service.retrieval_was_weak(deps) is False


class TestChoosingWhatToSuggest:
    @pytest.mark.anyio
    async def test_unindexed_links_are_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_links(monkeypatch, links("https://example.com/a"))
        suggestions = await service.suggest_pages("s", ["https://example.com/"])
        assert [suggestion.url for suggestion in suggestions] == [
            "https://example.com/a"
        ]
        assert suggestions[0].kind is SuggestionKind.ADD_PAGE
        assert suggestions[0].source_url == "https://example.com/"

    @pytest.mark.anyio
    async def test_a_link_already_indexed_is_not_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_links(monkeypatch, links("https://example.com/a"))
        suggestions = await service.suggest_pages(
            "s", ["https://example.com/", "https://example.com/a"]
        )
        assert suggestions == []

    @pytest.mark.anyio
    async def test_the_offer_is_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_links(
            monkeypatch, links(*[f"https://example.com/{n}" for n in range(20)])
        )
        suggestions = await service.suggest_pages("s", ["https://example.com/"])
        assert len(suggestions) == service.MAX_SUGGESTED_PAGES

    @pytest.mark.anyio
    async def test_the_offer_never_exceeds_the_slots_left(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Suggesting more than the session can hold sets up a partial failure."""
        install_links(
            monkeypatch, links(*[f"https://example.com/{n}" for n in range(20)])
        )
        held = [f"doc-{n}" for n in range(8)]
        suggestions = await service.suggest_pages("s", held)
        assert len(suggestions) == 2

    @pytest.mark.anyio
    async def test_a_full_session_is_offered_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_links(monkeypatch)
        held = [f"doc-{n}" for n in range(10)]
        assert await service.suggest_pages("s", held) == []

    @pytest.mark.anyio
    async def test_nothing_is_offered_without_a_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_links(monkeypatch)
        monkeypatch.setattr(service, "is_db_configured", lambda: False)
        assert await service.suggest_pages("s", []) == []


class StubVectorStore:
    def __init__(self, matches: list[RetrievedChunk] | None = None) -> None:
        self.matches = matches or []

    async def list_documents(self, session_id: str) -> list[str]:
        return ["https://example.com/"]

    async def search(
        self, session_id: str, query: str, limit: int = 4, min_score: float = 0.5
    ) -> list[RetrievedChunk]:
        return self.matches


def install_turn(
    monkeypatch: pytest.MonkeyPatch,
    refusal: str | None,
    searches: Sequence[SearchOutcome] = (),
) -> None:
    install_links(monkeypatch, links("https://example.com/a"))
    monkeypatch.setattr(service, "get_vector_store", lambda: StubVectorStore())

    deps = ChatDeps(
        session_id="s",
        documents=["https://example.com/"],
        searches=list(searches),
    )

    async def prepare(request: Any) -> Any:
        if refusal is not None:
            return service.RefusedTurn(
                intent="question",
                refusal=refusal,
                documents=["https://example.com/"],
            )
        return service.ReadyTurn(intent="question", history=[], deps=deps)

    async def fake_run(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(output="an answer", new_messages=lambda: [])

    class FakeStream:
        async def stream_output(self) -> Any:
            yield "an answer"

        def new_messages(self) -> list[Any]:
            return []

    class FakeStreamContext:
        async def __aenter__(self) -> FakeStream:
            return FakeStream()

        async def __aexit__(self, *exc: Any) -> None:
            return None

    def fake_run_stream(*args: Any, **kwargs: Any) -> FakeStreamContext:
        return FakeStreamContext()

    async def swallow(session_id: str, messages: Any) -> None:
        return None

    monkeypatch.setattr(service, "prepare_turn", prepare)
    monkeypatch.setattr(service, "append_turn", swallow)
    monkeypatch.setattr(
        service,
        "get_chat_agent",
        lambda: SimpleNamespace(run=fake_run, run_stream=fake_run_stream),
    )


class TestWhenTheOfferAppears:
    @pytest.mark.anyio
    async def test_a_gate_refusal_offers_more_pages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_turn(monkeypatch, refusal="None of them cover that.")
        response = await service.chat(
            ChatRequest(session_id="s", message="tell me about this file")
        )
        assert response.response == "None of them cover that."
        assert [s.url for s in response.suggestions] == ["https://example.com/a"]

    @pytest.mark.anyio
    async def test_an_empty_search_offers_more_pages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_turn(
            monkeypatch,
            refusal=None,
            searches=[SearchOutcome(query="links", matches=0, best_score=None)],
        )
        response = await service.chat(
            ChatRequest(session_id="s", message="what links are here")
        )
        assert response.response == "an answer"
        assert [s.url for s in response.suggestions] == ["https://example.com/a"]

    @pytest.mark.anyio
    async def test_a_good_answer_offers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trigger is the anti-nag policy: answers that land stay clean."""
        install_turn(
            monkeypatch,
            refusal=None,
            searches=[SearchOutcome(query="pricing", matches=3, best_score=0.8)],
        )
        response = await service.chat(
            ChatRequest(session_id="s", message="how do they price work")
        )
        assert response.suggestions == []

    @pytest.mark.anyio
    async def test_an_answer_with_no_search_offers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_turn(monkeypatch, refusal=None)
        response = await service.chat(
            ChatRequest(session_id="s", message="hello")
        )
        assert response.suggestions == []


class TestTheStreamingFrame:
    @pytest.mark.anyio
    async def test_a_refusal_ends_with_a_suggestions_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_turn(monkeypatch, refusal="None of them cover that.")
        events = [
            event
            async for event in service.stream_chat(
                ChatRequest(session_id="s", message="tell me about this file")
            )
        ]
        assert events[-1].type == "chat.suggestions"
        assert events[-1].suggestions is not None
        assert events[-1].suggestions[0].url == "https://example.com/a"

    @pytest.mark.anyio
    async def test_a_good_answer_sends_no_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_turn(
            monkeypatch,
            refusal=None,
            searches=[SearchOutcome(query="pricing", matches=3, best_score=0.8)],
        )
        events = [
            event
            async for event in service.stream_chat(
                ChatRequest(session_id="s", message="how do they price work")
            )
        ]
        assert all(event.type != "chat.suggestions" for event in events)


class TestAPausedRunIsAlreadyTheOffer:
    @pytest.mark.anyio
    async def test_a_pause_suppresses_the_passive_offer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both fire on the same turn — search finds nothing, then the model
        reaches for the tool — and they would prompt twice for one decision."""
        install_turn(
            monkeypatch,
            refusal=None,
            searches=[SearchOutcome(query="links", matches=0, best_score=None)],
        )

        async def park(
            session_id: str, intent: str, output: Any, messages: Any
        ) -> Any:
            return ChatResponse(
                session_id=session_id,
                response="Choose which to index.",
                intent=intent,
                pending_calls=[
                    PendingCallRead(
                        tool_call_id="call_1",
                        tool_name="add_url_to_knowledge_base",
                        kind=DeferredKind.CALL,
                    )
                ],
            )

        monkeypatch.setattr(service, "handle_output", park)
        response = await service.chat(
            ChatRequest(session_id="s", message="what links are here")
        )
        assert response.pending_calls
        assert response.suggestions == []
