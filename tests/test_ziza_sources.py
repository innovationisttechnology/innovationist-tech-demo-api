"""Tests for listing a session's sources.

The panel is rebuilt from this after a reload, so it has to report what the
session can actually answer from rather than what was once uploaded.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from app.ziza_chat import service
from app.ziza_chat.schemas import SourceKind
from app.ziza_chat.vector_store.store import StoredDocument


@pytest.fixture
def client() -> Any:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def stored(document: str, chunks: int = 3, day: int = 1) -> StoredDocument:
    return StoredDocument(
        document=document,
        chunks=chunks,
        added_at=datetime(2026, 9, day, tzinfo=timezone.utc),
    )


class StubStore:
    def __init__(self, documents: list[StoredDocument]) -> None:
        self.documents = documents

    async def summarize_documents(self, session_id: str) -> list[StoredDocument]:
        return self.documents


def install(monkeypatch: pytest.MonkeyPatch, documents: list[StoredDocument]) -> None:
    monkeypatch.setattr(service, "get_vector_store", lambda: StubStore(documents))
    monkeypatch.setattr(service, "is_db_configured", lambda: True)


class TestTellingFilesFromUrls:
    @pytest.mark.parametrize(
        "document,expected",
        [
            ("https://example.com/guide", SourceKind.URL),
            ("http://example.com", SourceKind.URL),
            ("handbook.pdf", SourceKind.FILE),
            ("notes.txt", SourceKind.FILE),
            # A filename can contain a colon without being a URL.
            ("meeting: 12 June.docx", SourceKind.FILE),
        ],
    )
    def test_the_document_name_is_the_only_signal(
        self, document: str, expected: SourceKind
    ) -> None:
        """URL-ingested documents are identified by their URL, files by their
        filename — nothing else distinguishes them."""
        assert service.source_kind(document) is expected


class TestListingThem:
    @pytest.mark.anyio
    async def test_every_document_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(
            monkeypatch,
            [stored("handbook.pdf", chunks=12), stored("https://example.com", 4, 2)],
        )
        response = await service.list_sources("s")
        assert [source.document for source in response.sources] == [
            "handbook.pdf",
            "https://example.com",
        ]
        assert [source.kind for source in response.sources] == [
            SourceKind.FILE,
            SourceKind.URL,
        ]
        assert [source.chunks for source in response.sources] == [12, 4]

    @pytest.mark.anyio
    async def test_capacity_is_reported_alongside(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So the panel can show remaining slots without a second call."""
        install(monkeypatch, [stored("a.pdf"), stored("b.pdf", day=2)])
        response = await service.list_sources("s")
        assert response.documents_used == 2
        assert response.documents_allowed > 2

    @pytest.mark.anyio
    async def test_an_empty_session_is_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, [])
        response = await service.list_sources("s")
        assert response.sources == []
        assert response.documents_used == 0

    @pytest.mark.anyio
    async def test_no_database_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, [stored("a.pdf")])
        monkeypatch.setattr(service, "is_db_configured", lambda: False)
        response = await service.list_sources("s")
        assert response.sources == []


class TestTheEndpoint:
    def test_it_reads_back_what_was_added(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.core.db import dependencies as db_dependencies
        from app.ziza_chat.schemas import KnowledgeSourceRead, KnowledgeSourcesResponse

        monkeypatch.setattr(db_dependencies, "is_db_configured", lambda: True)

        async def fake_list(session_id: str) -> KnowledgeSourcesResponse:
            return KnowledgeSourcesResponse(
                session_id=session_id,
                sources=[
                    KnowledgeSourceRead(
                        document="handbook.pdf",
                        kind=SourceKind.FILE,
                        chunks=9,
                        added_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                    )
                ],
                documents_used=1,
                documents_allowed=10,
            )

        monkeypatch.setattr(service, "list_sources", fake_list)
        body = client.get("/api/ziza/knowledge/session-1").json()
        assert body["sources"][0]["document"] == "handbook.pdf"
        assert body["sources"][0]["kind"] == "file"
        assert body["documents_used"] == 1
