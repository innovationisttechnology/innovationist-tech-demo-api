"""Tests for the opening questions offered when a document is added.

A visitor who has just uploaded a file has read none of it, so these are the
fastest route from "I added something" to "I know whether it holds what I
came for". They go through the same retrieval check the scope gate uses, so one
that reaches the visitor cannot be refused when clicked.
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
) -> list[tuple[str, str, list[StoredQuestion]]]:
    stored: list[tuple[str, str, list[StoredQuestion]]] = []

    async def write(document: str, text: str) -> list[str]:
        return list(written)

    async def store(
        session_id: str, document: str, questions: Sequence[StoredQuestion]
    ) -> None:
        stored.append((session_id, document, list(questions)))

    monkeypatch.setattr(service, "propose_starter_questions", write)
    monkeypatch.setattr(service, "store_starter_questions", store)
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


class StubStored:
    def __init__(self, document: str, questions: list[StoredQuestion]) -> None:
        self.document = document
        self.questions = questions


class TestReadingThemBack:
    @pytest.mark.anyio
    async def test_the_newest_document_is_the_one_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three uploads mean three questions about the newest, not nine."""

        async def load(session_id: str) -> Any:
            return StubStored(
                "third.pdf", [StoredQuestion(label="Q?", message="Q?")]
            )

        monkeypatch.setattr(service, "load_latest_starter_questions", load)
        response = await service.latest_starter_questions("s")
        assert response.document == "third.pdf"
        assert [s.message for s in response.suggestions] == ["Q?"]

    @pytest.mark.anyio
    async def test_nothing_stored_is_an_empty_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def load(session_id: str) -> Any:
            return None

        monkeypatch.setattr(service, "load_latest_starter_questions", load)
        response = await service.latest_starter_questions("s")
        assert response.suggestions == []
        assert response.document is None
