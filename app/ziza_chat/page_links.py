import logging
from datetime import datetime

from beanie import Document
from pydantic import BaseModel, Field
from pymongo import IndexModel

from app.core.config import settings
from app.core.utils.time import utc_now
from app.ziza_chat.document_loaders.web import CandidateLink

logger = logging.getLogger(__name__)


class StoredLink(BaseModel):
    url: str
    text: str


class PageLinks(Document):
    """Links found on a page when it was ingested.

    Kept so a suggestion can be made later without fetching the page again:
    the links are only in the raw HTML, which nothing holds on to once the
    text has been chunked.
    """

    session_id: str
    document: str
    links: list[StoredLink] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "page_links"
        indexes = [
            IndexModel(
                [("session_id", 1), ("document", 1)],
                unique=True,
                name="uniq_session_document_links",
            ),
            IndexModel(
                [("created_at", 1)],
                name="ttl_page_links",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]


async def store_page_links(
    session_id: str, document: str, links: list[CandidateLink]
) -> None:
    stored = [StoredLink(url=link.url, text=link.text) for link in links]
    try:
        existing = await PageLinks.find_one(
            PageLinks.session_id == session_id, PageLinks.document == document
        )
        if existing is None:
            await PageLinks(
                session_id=session_id, document=document, links=stored
            ).insert()
            return
        existing.links = stored
        await existing.save()
    except Exception as failure:
        logger.warning("Could not store links for %s: %s", document, failure)


async def load_page_links(session_id: str) -> list[PageLinks]:
    return await PageLinks.find(PageLinks.session_id == session_id).to_list()


async def clear_page_links(session_id: str) -> int:
    result = await PageLinks.find(PageLinks.session_id == session_id).delete()
    return int(result.deleted_count) if result else 0
