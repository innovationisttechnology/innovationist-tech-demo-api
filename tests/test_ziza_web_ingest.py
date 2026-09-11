"""Tests for URL ingestion — chiefly the SSRF surface.

Fetching a user-supplied URL makes the server issue requests on their behalf,
so most of these assert what is *refused* rather than what is fetched.
"""

from typing import Any

import pytest

from app.ziza_chat import service
from app.ziza_chat.agents.page_summary import PageSummary
from app.ziza_chat.document_loaders.base import UnsupportedDocumentError
from app.ziza_chat.document_loaders.web import (
    MAX_CANDIDATE_LINKS,
    FetchedPage,
    UnsafeUrlError,
    assert_public_url,
    extract_links,
    extract_title,
    html_to_text,
)
from app.ziza_chat.schemas import (
    KnowledgeIngestResponse,
    UrlLinkSelectionRequest,
)


class TestUrlSafety:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8181/api/ziza/chat",
            "http://127.0.0.1/",
            "http://[::1]/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/admin",
            "http://172.16.0.1/",
            "http://0.0.0.0/",
        ],
    )
    def test_internal_addresses_are_refused(self, url: str) -> None:
        with pytest.raises(UnsafeUrlError):
            assert_public_url(url)

    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/"],
    )
    def test_non_http_schemes_are_refused(self, url: str) -> None:
        with pytest.raises(UnsafeUrlError) as failure:
            assert_public_url(url)
        assert "http" in str(failure.value)

    def test_url_without_a_hostname_is_refused(self) -> None:
        with pytest.raises(UnsafeUrlError):
            assert_public_url("http:///nohost")

    def test_unresolvable_hostname_is_refused(self) -> None:
        with pytest.raises(UnsafeUrlError):
            assert_public_url("http://this-host-does-not-exist.invalid/")

    def test_hostname_resolving_to_loopback_is_refused(self) -> None:
        # The check must act on the resolved address, not on how the name looks
        # — this is the shape a DNS-based bypass takes.
        with pytest.raises(UnsafeUrlError) as failure:
            assert_public_url("http://localtest.me/")
        assert "non-public" in str(failure.value)


class TestHtmlExtraction:
    def test_strips_navigation_and_scripts(self) -> None:
        html = b"""
        <html><head><title>Platform Docs</title></head>
        <body>
          <nav>Home About Contact</nav>
          <script>trackEverything()</script>
          <style>.a{color:red}</style>
          <h1>Deployment pipeline</h1>
          <p>Releases ship on Tuesdays and Thursdays.</p>
          <footer>Copyright 2026</footer>
        </body></html>
        """
        text = html_to_text(html)
        assert "Deployment pipeline" in text
        assert "Releases ship on Tuesdays" in text
        assert "trackEverything" not in text
        assert "Home About Contact" not in text
        assert "Copyright 2026" not in text

    def test_keeps_list_and_table_content(self) -> None:
        html = b"""
        <html><body>
          <ul><li>Sev-1 incidents: 7</li></ul>
          <table><tr><td>content-sync</td><td>68%</td></tr></table>
        </body></html>
        """
        text = html_to_text(html)
        assert "Sev-1 incidents: 7" in text
        assert "content-sync" in text
        assert "68%" in text

    def test_falls_back_when_there_are_no_block_elements(self) -> None:
        text = html_to_text(b"<html><body>bare text with no tags</body></html>")
        assert "bare text with no tags" in text

    def test_title_extracted_from_html_only(self) -> None:
        assert extract_title(b"<html><title> Docs </title></html>", "text/html") == "Docs"
        assert extract_title(b"%PDF-1.7", "application/pdf") == ""


class TestPageSummary:
    def test_search_text_includes_populated_fields(self) -> None:
        summary = PageSummary(
            summary="Reference for the deployment pipeline.",
            topics=["deployments", "release cadence"],
            entities=["platform team"],
            key_facts=["Releases ship Tuesdays and Thursdays"],
        )
        text = summary.to_search_text()
        assert "Reference for the deployment pipeline." in text
        assert "release cadence" in text
        assert "platform team" in text
        assert "Releases ship Tuesdays and Thursdays" in text

    def test_empty_fields_are_omitted(self) -> None:
        summary = PageSummary(summary="A login wall with no content.")
        assert summary.to_search_text() == "A login wall with no content."


class TestExtractingLinks:
    """Candidates offered for a deferred add_url_to_knowledge_base call."""

    def test_relative_links_resolve_against_the_page(self) -> None:
        page = b'<html><body><a href="/docs/setup">Setup</a></body></html>'
        links = extract_links(page, "https://example.com/guide")
        assert [link.url for link in links] == ["https://example.com/docs/setup"]
        assert links[0].text == "Setup"

    def test_links_in_navigation_are_kept(self) -> None:
        """html_to_text strips nav, which is where a site's own links live."""
        page = b'<html><nav><a href="/api">API</a></nav></html>'
        links = extract_links(page, "https://example.com/guide")
        assert [link.url for link in links] == ["https://example.com/api"]

    def test_other_sites_are_left_out(self) -> None:
        page = (
            b'<html><a href="https://elsewhere.example.net/x">Off site</a>'
            b'<a href="/mine">Mine</a></html>'
        )
        links = extract_links(page, "https://example.com/guide")
        assert [link.url for link in links] == ["https://example.com/mine"]

    def test_non_http_schemes_are_left_out(self) -> None:
        page = (
            b'<html><a href="mailto:hi@example.com">Mail</a>'
            b'<a href="javascript:alert(1)">Click</a></html>'
        )
        assert extract_links(page, "https://example.com/guide") == []

    def test_fragments_and_duplicates_collapse(self) -> None:
        page = (
            b'<html><a href="/a">One</a><a href="/a#top">Again</a>'
            b'<a href="/a/">Trailing</a></html>'
        )
        links = extract_links(page, "https://example.com/guide")
        assert [link.url for link in links] == ["https://example.com/a"]

    def test_the_page_itself_is_not_offered(self) -> None:
        page = b'<html><a href="/guide">This page</a></html>'
        assert extract_links(page, "https://example.com/guide") == []

    def test_the_offer_is_capped(self) -> None:
        anchors = b"".join(
            f'<a href="/page-{index}">Page {index}</a>'.encode() for index in range(40)
        )
        links = extract_links(b"<html>" + anchors + b"</html>", "https://example.com/")
        assert len(links) == MAX_CANDIDATE_LINKS

    def test_a_link_with_no_text_falls_back_to_its_url(self) -> None:
        page = b'<html><a href="/icon"><img src="i.png"></a></html>'
        links = extract_links(page, "https://example.com/guide")
        assert links[0].text == "https://example.com/icon"


class StubVectorStore:
    def __init__(self) -> None:
        self.sections: list[Any] = []

    async def list_documents(self, session_id: str) -> list[str]:
        return []

    async def count_documents(self, session_id: str) -> int:
        return 1

    async def add_sections(self, session_id: str, sections: Any) -> int:
        self.sections.extend(sections)
        return len(list(sections))

    async def wait_until_searchable(self, session_id: str) -> bool:
        return True


def install_page(monkeypatch: pytest.MonkeyPatch, body: bytes) -> StubVectorStore:
    store = StubVectorStore()

    async def fetch(url: str) -> FetchedPage:
        return FetchedPage(
            url=url, title="The Guide", content_type="text/html", body=body
        )

    async def no_summary(title: str, url: str, text: str) -> Any:
        raise RuntimeError("summariser off in tests")

    monkeypatch.setattr(service, "fetch_page", fetch)
    monkeypatch.setattr(service, "summarize_page", no_summary)
    monkeypatch.setattr(service, "get_vector_store", lambda: store)
    return store


PAGE_WITH_LINKS = (
    b"<html><body><p>Some readable body text here.</p>"
    b'<a href="/setup">Setup</a><a href="https://elsewhere.example.net/x">Off</a>'
    b"</body></html>"
)


class TestTheLinkOffer:
    @pytest.mark.anyio
    async def test_ingesting_a_page_offers_its_links(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_page(monkeypatch, PAGE_WITH_LINKS)
        result = await service.ingest_url("session-1", "https://example.com/guide")
        assert [link.url for link in result.candidate_links] == [
            "https://example.com/setup"
        ]

    @pytest.mark.anyio
    async def test_the_page_is_indexed_without_waiting_for_a_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pasting a URL is an unambiguous 'index this'; only the links are open."""
        store = install_page(monkeypatch, PAGE_WITH_LINKS)
        result = await service.ingest_url("session-1", "https://example.com/guide")
        assert result.chunks_ingested == 1
        assert store.sections

    @pytest.mark.anyio
    async def test_a_selected_link_is_not_itself_an_offer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One level deep — indexing a link must not open another prompt."""
        install_page(monkeypatch, PAGE_WITH_LINKS)
        result = await service.ingest_url(
            "session-1", "https://example.com/setup", offer_links=False
        )
        assert result.candidate_links == []


class TestIngestingTheSelection:
    @pytest.mark.anyio
    async def test_the_selection_is_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexed: list[str] = []

        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            indexed.append(url)
            assert offer_links is False
            return KnowledgeIngestResponse(
                session_id=session_id, source=url, chunks_ingested=2, searchable=True
            )

        monkeypatch.setattr(service, "ingest_url", fake_ingest)
        monkeypatch.setattr(service, "get_vector_store", lambda: StubVectorStore())
        response = await service.ingest_offered_links(
            UrlLinkSelectionRequest(
                session_id="session-1",
                url="https://example.com/guide",
                selected_links=[
                    "https://example.com/setup",
                    "https://example.com/api",
                ],
            )
        )
        assert indexed == ["https://example.com/setup", "https://example.com/api"]
        assert len(response.indexed) == 2
        assert response.failed == []

    @pytest.mark.anyio
    async def test_an_offsite_link_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a 'selection' walks the server around the web."""
        indexed: list[str] = []

        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            indexed.append(url)
            raise AssertionError("should never be reached")

        monkeypatch.setattr(service, "ingest_url", fake_ingest)
        with pytest.raises(service.UnofferedLinkError):
            await service.ingest_offered_links(
                UrlLinkSelectionRequest(
                    session_id="session-1",
                    url="https://example.com/guide",
                    selected_links=["https://evil.example.net/steal"],
                )
            )
        assert indexed == []

    @pytest.mark.anyio
    async def test_one_bad_link_does_not_lose_the_rest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_ingest(
            session_id: str, url: str, offer_links: bool = True
        ) -> Any:
            if url.endswith("/setup"):
                raise UnsupportedDocumentError("no readable text")
            return KnowledgeIngestResponse(
                session_id=session_id, source=url, chunks_ingested=2, searchable=True
            )

        monkeypatch.setattr(service, "ingest_url", fake_ingest)
        monkeypatch.setattr(service, "get_vector_store", lambda: StubVectorStore())
        response = await service.ingest_offered_links(
            UrlLinkSelectionRequest(
                session_id="session-1",
                url="https://example.com/guide",
                selected_links=[
                    "https://example.com/setup",
                    "https://example.com/api",
                ],
            )
        )
        assert [result.source for result in response.indexed] == [
            "https://example.com/api"
        ]
        assert [failure.url for failure in response.failed] == [
            "https://example.com/setup"
        ]
