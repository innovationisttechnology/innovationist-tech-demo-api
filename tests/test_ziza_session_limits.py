"""Tests for the per-session document limit.

The store is stubbed so these run without MongoDB; what matters is the decision
`assert_capacity` makes, and that it is made before any expensive work.
"""

import pytest

from app.ziza_chat import service
from app.ziza_chat.config import ziza_settings


class StubStore:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    async def list_documents(self, session_id: str) -> list[str]:
        return self.documents


def use_documents(monkeypatch: pytest.MonkeyPatch, documents: list[str]) -> None:
    monkeypatch.setattr(service, "get_vector_store", lambda: StubStore(documents))


class TestAssertCapacity:
    @pytest.mark.anyio
    async def test_allows_a_new_document_below_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        use_documents(monkeypatch, ["a.pdf", "b.pdf"])
        await service.assert_capacity("session-1", "c.pdf")

    @pytest.mark.anyio
    async def test_refuses_a_new_document_at_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        full = [f"doc{index}.pdf" for index in range(
            ziza_settings.max_documents_per_session
        )]
        use_documents(monkeypatch, full)
        with pytest.raises(service.SessionLimitError) as failure:
            await service.assert_capacity("session-1", "one-too-many.pdf")
        assert str(ziza_settings.max_documents_per_session) in str(failure.value)

    @pytest.mark.anyio
    async def test_reingesting_an_existing_document_is_allowed_when_full(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-uploading occupies a slot it has already paid for, so a full
        # session can still refresh what it holds.
        full = [f"doc{index}.pdf" for index in range(
            ziza_settings.max_documents_per_session
        )]
        use_documents(monkeypatch, full)
        await service.assert_capacity("session-1", "doc0.pdf")

    @pytest.mark.anyio
    async def test_empty_session_accepts_anything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        use_documents(monkeypatch, [])
        await service.assert_capacity("session-1", "first.pdf")


class TestLimitIsCheckedBeforeExpensiveWork:
    @pytest.mark.anyio
    async def test_rejected_upload_never_parses_or_captions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal must cost nothing — no extraction, no model calls."""
        full = [f"doc{index}.pdf" for index in range(
            ziza_settings.max_documents_per_session
        )]
        use_documents(monkeypatch, full)

        def must_not_run(*args: object, **kwargs: object) -> object:
            raise AssertionError("document parsed despite a full session")

        monkeypatch.setattr(service, "load_document", must_not_run)

        with pytest.raises(service.SessionLimitError):
            await service.ingest_file(
                session_id="session-1",
                filename="new.pdf",
                content_type="application/pdf",
                data=b"%PDF-1.7",
            )

    @pytest.mark.anyio
    async def test_rejected_url_is_never_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        full = [f"doc{index}.pdf" for index in range(
            ziza_settings.max_documents_per_session
        )]
        use_documents(monkeypatch, full)

        async def must_not_fetch(url: str) -> object:
            raise AssertionError("URL fetched despite a full session")

        monkeypatch.setattr(service, "fetch_page", must_not_fetch)

        with pytest.raises(service.SessionLimitError):
            await service.ingest_url("session-1", "https://example.com/page")
