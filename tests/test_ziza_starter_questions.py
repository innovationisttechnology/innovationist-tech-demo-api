"""Tests for the opening questions offered when a document is added.

A visitor who has just uploaded a file has read none of it, so these are the
fastest route from "I added something" to "I know whether it holds what I came
for". They are a cold start and nothing else: offered once, retired the moment
the visitor asks anything, and never offered again in that session.
"""

from typing import Any, Sequence

import pytest
from pydantic_ai.models.test import TestModel

from app.ziza_chat import service
from app.ziza_chat.agents.starter_questions import (
    MAX_STARTER_QUESTIONS,
    get_starter_question_agent,
    propose_starter_questions,
)
from app.ziza_chat.schemas import SuggestionKind
from app.ziza_chat.starter_questions import StoredQuestion
from app.ziza_chat.vector_store.store import RetrievedChunk


class StubStore:
    def __init__(self, answerable: Sequence[str] = ()) -> None:
        self.answerable = set(answerable)

    async def search(
        self, session_id: str, query: str, limit: int = 4, min_score: float = 0.5
    ) -> list[RetrievedChunk]:
        if query not in self.answerable:
            return []
        return [RetrievedChunk(text="a passage", source="handbook.pdf", score=0.7)]


def install(
    monkeypatch: pytest.MonkeyPatch,
    written: Sequence[str],
    answerable: Sequence[str] | None = None,
    wanted: bool = True,
) -> list[tuple[str, str, list[StoredQuestion]]]:
    stored: list[tuple[str, str, list[StoredQuestion]]] = []

    async def write(document: str, text: str) -> list[str]:
        return list(written)

    async def store(
        session_id: str, document: str, questions: Sequence[StoredQuestion]
    ) -> None:
        stored.append((session_id, document, list(questions)))

    async def are_wanted(session_id: str) -> bool:
        return wanted

    monkeypatch.setattr(service, "propose_starter_questions", write)
    monkeypatch.setattr(service, "store_starter_questions", store)
    monkeypatch.setattr(service, "starters_are_wanted", are_wanted)
    monkeypatch.setattr(
        service,
        "get_vector_store",
        lambda: StubStore(written if answerable is None else answerable),
    )
    monkeypatch.setattr(service, "is_db_configured", lambda: True)
    return stored


class TestWritingThem:
    @pytest.mark.anyio
    async def test_the_questions_become_suggestions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, ["What is the release cadence?"])
        offered = await service.starter_questions("s", "handbook.pdf", "text")
        assert [s.message for s in offered] == ["What is the release cadence?"]
        assert offered[0].kind is SuggestionKind.ASK
        # Clicking one sends it as an ordinary chat message.
        assert offered[0].label == offered[0].message

    @pytest.mark.anyio
    async def test_one_the_gate_would_refuse_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Written from the raw text, but the chunked copy has to answer it."""
        install(monkeypatch, ["Covered?", "Not covered?"], answerable=["Covered?"])
        offered = await service.starter_questions("s", "handbook.pdf", "text")
        assert [s.message for s in offered] == ["Covered?"]

    @pytest.mark.anyio
    async def test_a_document_with_nothing_to_ask_offers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, [])
        assert await service.starter_questions("s", "login.html", "text") == []

    @pytest.mark.anyio
    async def test_only_the_answerable_ones_are_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = install(
            monkeypatch, ["Covered?", "Not covered?"], answerable=["Covered?"]
        )
        await service.starter_questions("s", "handbook.pdf", "text")
        assert stored == [("s", "handbook.pdf", [StoredQuestion(
            label="Covered?", message="Covered?"
        )])]

    @pytest.mark.anyio
    async def test_a_failure_costs_the_questions_not_the_upload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The document is already indexed by the time these are written."""
        install(monkeypatch, [])

        async def explode(document: str, text: str) -> list[str]:
            raise RuntimeError("provider is down")

        monkeypatch.setattr(service, "propose_starter_questions", explode)
        assert await service.starter_questions("s", "handbook.pdf", "text") == []


class TestTheWriter:
    @pytest.mark.anyio
    async def test_blank_questions_are_discarded(self) -> None:
        with get_starter_question_agent().override(
            model=TestModel(custom_output_args={"questions": ["  ", "Real?"]})
        ):
            assert await propose_starter_questions("d", "text") == ["Real?"]

    def test_the_schema_caps_the_count(self) -> None:
        """The cap is in the output schema, so the model is told, not trimmed."""
        from app.ziza_chat.agents.starter_questions import StarterQuestionList

        assert (
            StarterQuestionList.model_fields["questions"].metadata[0].max_length
            == MAX_STARTER_QUESTIONS
        )


class StubStarters:
    def __init__(
        self,
        document: str | None,
        questions: list[StoredQuestion],
        conversation_started: bool = False,
    ) -> None:
        self.document = document
        self.questions = questions
        self.conversation_started = conversation_started


def install_stored(monkeypatch: pytest.MonkeyPatch, stored: Any) -> None:
    async def load(session_id: str) -> Any:
        return stored

    monkeypatch.setattr(service, "load_starters", load)


class TestOfferingThemOnlyOnce:
    @pytest.mark.anyio
    async def test_a_session_already_in_conversation_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """And costs no model call — the check happens before the writer."""
        called: list[str] = []
        stored = install(monkeypatch, ["Q?"], wanted=False)

        async def must_not_run(document: str, text: str) -> list[str]:
            called.append(document)
            return ["Q?"]

        monkeypatch.setattr(service, "propose_starter_questions", must_not_run)
        assert await service.starter_questions("s", "second.pdf", "text") == []
        assert called == []
        assert stored == []

    @pytest.mark.anyio
    async def test_no_database_offers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, ["Q?"])
        monkeypatch.setattr(service, "is_db_configured", lambda: False)
        assert await service.starter_questions("s", "handbook.pdf", "text") == []

    @pytest.mark.anyio
    async def test_asking_anything_retires_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retired: list[str] = []

        async def mark(session_id: str) -> None:
            retired.append(session_id)

        monkeypatch.setattr(service, "mark_conversation_started", mark)
        monkeypatch.setattr(service, "is_db_configured", lambda: True)
        await service.retire_starters("session-1")
        assert retired == ["session-1"]

    @pytest.mark.anyio
    async def test_retiring_is_skipped_without_a_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def must_not_run(session_id: str) -> None:
            raise AssertionError("touched the database when there is none")

        monkeypatch.setattr(service, "mark_conversation_started", must_not_run)
        monkeypatch.setattr(service, "is_db_configured", lambda: False)
        await service.retire_starters("session-1")


class TestReadingThemBack:
    @pytest.mark.anyio
    async def test_the_newest_document_is_the_one_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two uploads before asking mean three questions about the newer."""
        install_stored(
            monkeypatch,
            StubStarters("second.pdf", [StoredQuestion(label="Q?", message="Q?")]),
        )
        response = await service.latest_starter_questions("s")
        assert response.document == "second.pdf"
        assert [s.message for s in response.suggestions] == ["Q?"]

    @pytest.mark.anyio
    async def test_nothing_stored_is_an_empty_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_stored(monkeypatch, None)
        response = await service.latest_starter_questions("s")
        assert response.suggestions == []
        assert response.document is None

    @pytest.mark.anyio
    async def test_a_started_conversation_reads_back_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Belt and braces: retiring clears the questions, and the read refuses
        to serve them even if a stale row still held some."""
        install_stored(
            monkeypatch,
            StubStarters(
                "handbook.pdf",
                [StoredQuestion(label="Q?", message="Q?")],
                conversation_started=True,
            ),
        )
        response = await service.latest_starter_questions("s")
        assert response.suggestions == []
        assert response.document is None
