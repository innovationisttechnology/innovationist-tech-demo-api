"""Tests for URL ingestion — chiefly the SSRF surface.

Fetching a user-supplied URL makes the server issue requests on their behalf,
so most of these assert what is *refused* rather than what is fetched.
"""

import pytest

from app.ziza_chat.agents.page_summary import PageSummary
from app.ziza_chat.document_loaders.web import (
    UnsafeUrlError,
    assert_public_url,
    extract_title,
    html_to_text,
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
