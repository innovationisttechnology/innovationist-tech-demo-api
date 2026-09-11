"""Fetching and extracting user-supplied URLs.

Fetching a URL the user chose means the server makes a request on their behalf,
which is the classic SSRF shape: without checks, a submitted
`http://169.254.169.254/` pulls cloud credentials into the knowledge base and
`http://10.0.0.5/` reaches anything on the private network. Every hop is
resolved and checked against private address space before a connection is made.
"""

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from app.ziza_chat.document_loaders.base import UnsupportedDocumentError

logger = logging.getLogger(__name__)

MAX_PAGE_BYTES = 10 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 15.0
MAX_REDIRECTS = 5

USER_AGENT = "ZizaBot/1.0 (+https://demo.innovationisttech.com)"

STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "svg",
    "iframe",
)


class UnsafeUrlError(ValueError):
    pass


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str
    content_type: str
    body: bytes


def assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(
            f"Only http and https URLs can be fetched (got {parsed.scheme or 'none'!r})."
        )
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeUrlError("URL has no hostname.")

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or 0, type=socket.SOCK_STREAM)
    except socket.gaierror as failure:
        raise UnsafeUrlError(f"Could not resolve {hostname!r}.") from failure

    for *_, sockaddr in addresses:
        address = ipaddress.ip_address(sockaddr[0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise UnsafeUrlError(
                f"{hostname!r} resolves to a non-public address ({address}); "
                "internal hosts cannot be fetched."
            )


async def fetch_page(url: str) -> FetchedPage:
    """Fetch a URL, validating the destination at every redirect hop."""
    current_url = url
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=FETCH_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            assert_public_url(current_url)
            response = await client.get(current_url)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise UnsupportedDocumentError(f"{current_url} redirected nowhere.")
                # Re-validated on the next pass — a public URL is allowed to
                # redirect to an internal one, which is how naive fetchers leak.
                current_url = str(response.url.join(location))
                continue

            if response.status_code >= 400:
                raise UnsupportedDocumentError(
                    f"{current_url} returned HTTP {response.status_code}."
                )
            body = response.content
            if len(body) > MAX_PAGE_BYTES:
                raise UnsupportedDocumentError(
                    f"{current_url} is larger than "
                    f"{MAX_PAGE_BYTES // (1024 * 1024)}MB."
                )
            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            return FetchedPage(
                url=current_url,
                title=extract_title(body, content_type),
                content_type=content_type or "application/octet-stream",
                body=body,
            )

    raise UnsupportedDocumentError(f"{url} exceeded {MAX_REDIRECTS} redirects.")


def extract_title(body: bytes, content_type: str) -> str:
    if "html" not in content_type:
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def html_to_text(body: bytes) -> str:
    """Readable page text with navigation, scripts, and chrome removed.

    Boilerplate that repeats on every page of a site is actively harmful in a
    retrieval index: it dilutes chunks and makes unrelated pages look similar.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body, "html.parser")
    for tag in soup(list(STRIP_TAGS)):
        tag.decompose()

    blocks: list[str] = []
    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "pre", "blockquote"]
    ):
        text = " ".join(element.get_text(" ", strip=True).split())
        if text and text not in blocks[-3:]:
            blocks.append(text)

    if not blocks:
        text = " ".join(soup.get_text(" ", strip=True).split())
        return text
    return "\n\n".join(blocks)


MAX_CANDIDATE_LINKS = 10


@dataclass(frozen=True)
class CandidateLink:
    url: str
    text: str


def extract_links(
    body: bytes, base_url: str, limit: int = MAX_CANDIDATE_LINKS
) -> list[CandidateLink]:
    """Same-origin links found on a page, in document order.

    Read from the raw HTML rather than from html_to_text output: that strips
    nav, header, footer, and aside, which is exactly where a site keeps the
    links to the rest of itself. Same-origin only, because offering a visitor
    every outbound link on a page turns one request into an open crawl.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body, "html.parser")
    origin = urlparse(base_url).netloc
    already_offered = {base_url.rstrip("/")}
    candidates: list[CandidateLink] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        resolved = urlparse(urljoin(base_url, href))
        if resolved.scheme not in ("http", "https") or resolved.netloc != origin:
            continue
        normalized = resolved._replace(fragment="").geturl()
        if normalized.rstrip("/") in already_offered:
            continue
        already_offered.add(normalized.rstrip("/"))
        label = " ".join(anchor.get_text(" ", strip=True).split())
        candidates.append(CandidateLink(url=normalized, text=label[:120] or normalized))
        if len(candidates) >= limit:
            break

    return candidates
