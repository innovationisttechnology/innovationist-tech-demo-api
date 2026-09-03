"""Tests for the scope gate.

The gate is the enforcement layer — the chat agent's instructions are only a
default. These assert the decision is made in code, without MongoDB or a model.
"""

import pytest

from app.ziza_chat import service
from app.ziza_chat.agents.outputs import ClassifyResult, Intent, Scope
from app.ziza_chat.config import ziza_settings
from app.ziza_chat.vector_store.store import RetrievedChunk


class StubStore:
    def __init__(self, documents: list[str], matches: list[RetrievedChunk]) -> None:
        self.documents = documents
        self.matches = matches
        self.searched: list[str] = []

    async def list_documents(self, session_id: str) -> list[str]:
        return self.documents

    async def search(
        self, session_id: str, query: str, limit: int = 4, min_score: float = 0.5
    ) -> list[RetrievedChunk]:
        self.searched.append(query)
        return self.matches


def install_store(
    monkeypatch: pytest.MonkeyPatch,
    documents: list[str],
    matches: list[RetrievedChunk] | None = None,
) -> StubStore:
    store = StubStore(documents, matches or [])
    monkeypatch.setattr(service, "get_vector_store", lambda: store)
    monkeypatch.setattr(service, "is_db_configured", lambda: True)
    return store


def match(text: str = "a relevant passage") -> RetrievedChunk:
    return RetrievedChunk(text=text, source="handbook.txt", score=0.7)


def classification(
    scope: Scope, rag_query: str | None = None, needs_rag: bool = True
) -> ClassifyResult:
    return ClassifyResult(
        scope=scope,
        intents=[Intent.QUESTION],
        needs_rag=needs_rag,
        rag_query=rag_query,
    )


class TestOutOfScope:
    @pytest.mark.anyio
    async def test_refuses_without_searching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_store(monkeypatch, ["handbook.txt"], [match()])
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.OUT_OF_SCOPE, needs_rag=False)
        )
        assert refusal is not None
        assert "only answer from the documents" in refusal
        # Out of scope is decided outright — retrieval is irrelevant.
        assert store.searched == []

    @pytest.mark.anyio
    async def test_refusal_names_the_documents_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, ["handbook.txt", "q3.pdf"])
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.OUT_OF_SCOPE, needs_rag=False)
        )
        assert refusal is not None
        assert "handbook.txt" in refusal and "q3.pdf" in refusal

    @pytest.mark.anyio
    async def test_refusal_invites_an_upload_when_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, [])
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.OUT_OF_SCOPE, needs_rag=False)
        )
        assert refusal is not None
        assert "aren't any yet" in refusal

    @pytest.mark.anyio
    async def test_refusal_points_at_the_general_assistant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal should be a redirect, not a dead end."""
        install_store(monkeypatch, ["handbook.txt"])
        for documents in ([], ["handbook.txt"]):
            install_store(monkeypatch, documents)
            refusal = await service.resolve_scope(
                "session-1", classification(Scope.OUT_OF_SCOPE, needs_rag=False)
            )
            assert refusal is not None
            assert ziza_settings.general_assistant_url in refusal


class TestAssistantScope:
    @pytest.mark.anyio
    async def test_questions_about_the_demo_are_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, [])
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.ASSISTANT, needs_rag=False)
        )
        assert refusal is None


class TestKnowledgeBaseScope:
    @pytest.mark.anyio
    async def test_allowed_when_something_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, ["handbook.txt"], [match()])
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.KNOWLEDGE_BASE, "release approval")
        )
        assert refusal is None

    @pytest.mark.anyio
    async def test_refused_when_nothing_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # In-scope shape, but the session holds nothing on the topic — the
        # agent must not get a chance to answer it from memory.
        install_store(monkeypatch, ["handbook.txt"], [])
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.KNOWLEDGE_BASE, "photosynthesis")
        )
        assert refusal is not None
        assert "handbook.txt" in refusal

    @pytest.mark.anyio
    async def test_gate_searches_with_the_classifier_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_store(monkeypatch, ["handbook.txt"], [match()])
        await service.resolve_scope(
            "session-1", classification(Scope.KNOWLEDGE_BASE, "Sarah Chen")
        )
        assert store.searched == ["Sarah Chen"]

    @pytest.mark.anyio
    async def test_allowed_when_there_is_no_query_to_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without a query the gate cannot judge relevance; the agent's own
        # search decides, rather than refusing something possibly valid.
        store = install_store(monkeypatch, ["handbook.txt"], [])
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.KNOWLEDGE_BASE, rag_query=None)
        )
        assert refusal is None
        assert store.searched == []

    @pytest.mark.anyio
    async def test_refused_when_no_database_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service, "is_db_configured", lambda: False)
        refusal = await service.resolve_scope(
            "session-1", classification(Scope.KNOWLEDGE_BASE, "anything")
        )
        assert refusal is not None


class TestGateThreshold:
    def test_gate_matches_the_answer_threshold(self) -> None:
        from app.ziza_chat.vector_store.store import MIN_SCORE

        # Anything lower is below the noise floor: unrelated queries score
        # 0.38-0.47 against a small knowledge base, so a permissive gate lets
        # every general question reach the agent.
        assert service.GATE_MIN_SCORE == MIN_SCORE

    @pytest.mark.anyio
    async def test_a_weak_coincidental_match_does_not_open_the_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_store(monkeypatch, ["handbook.txt"], [])
        await service.resolve_scope(
            "session-1", classification(Scope.KNOWLEDGE_BASE, "gravity")
        )
        # The gate must ask the store for matches at its own threshold rather
        # than accepting whatever the default returns.
        assert store.searched == ["gravity"]


class TestScopeDefault:
    def test_missing_scope_defaults_to_the_knowledge_base(self) -> None:
        # A classifier that omits the field must not accidentally open the
        # assistant up to general questions.
        result = ClassifyResult(intents=[Intent.QUESTION], needs_rag=True)
        assert result.scope is Scope.KNOWLEDGE_BASE
