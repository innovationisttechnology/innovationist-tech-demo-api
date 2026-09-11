import logging

import httpx
from pydantic_ai import ApprovalRequired, CallDeferred, RunContext

from app.ziza_chat.config import ziza_settings
from app.ziza_chat.deps import ChatDeps
from app.ziza_chat.document_loaders.base import UnsupportedDocumentError
from app.ziza_chat.document_loaders.web import (
    UnsafeUrlError,
    extract_links,
    fetch_page,
)
from app.ziza_chat.knowledge_base import clear_knowledge, describe_holdings
from app.ziza_chat.tool_logging import log_step

logger = logging.getLogger(__name__)


async def clear_knowledge_base(context: RunContext[ChatDeps]) -> str:
    """Delete every document in this session's knowledge base.

    Call this when the visitor asks to clear, reset, delete, or start over with
    their uploaded material. It removes every document, every image
    description, and the record of this conversation.

    Call it as soon as they ask. Do not ask them to confirm first and do not
    describe what would happen instead of calling it — confirming the deletion
    is handled for you, and nothing is removed until it is confirmed. Only call
    it when they have actually asked; never to tidy up on your own initiative.
    """
    session_id = context.deps.session_id
    if not context.tool_call_approved:
        documents = await describe_holdings(session_id)
        log_step(
            session_id, "defer", f"approval needed to clear {len(documents)} document(s)"
        )
        raise ApprovalRequired(
            metadata={
                "action": "clear_knowledge_base",
                "documents": documents,
                "summary": (
                    f"Delete {len(documents)} document(s) from this session, "
                    "along with their image descriptions and this conversation."
                ),
            }
        )
    chunks_deleted = await clear_knowledge(session_id)
    log_step(session_id, "cleared", f"{chunks_deleted} passage(s) deleted")
    return (
        f"Deleted {chunks_deleted} passage(s). This session's knowledge base is "
        "now empty."
    )


async def add_url_to_knowledge_base(context: RunContext[ChatDeps], url: str) -> str:
    """Add a web page to this session's knowledge base so it can be searched.

    Call this when the visitor gives you a link and asks you to read, add,
    index, or answer from it. Pass the URL exactly as they wrote it. Also call
    it when they ask what else a page in this session links to, since those
    links are not part of the indexed text.

    This does not index anything on its own. It reads the page, finds the other
    pages that page links to, and hands the visitor the list so they can pick
    which of them to include alongside it — indexing happens once they have
    chosen. So do not ask them which links they want or whether to go ahead:
    that choice is put to them for you, and nothing is added until they make
    it. Report what you were told was indexed afterwards, and do not claim a
    page is searchable before then.
    """
    session_id = context.deps.session_id
    log_step(session_id, "add-url", url)
    slots_left = ziza_settings.max_documents_per_session - len(context.deps.documents)
    if url not in context.deps.documents and slots_left <= 0:
        log_step(session_id, "add-url", f"refused — {slots_left} slot(s) left")
        return (
            f"This session already holds its limit of "
            f"{ziza_settings.max_documents_per_session} documents, so nothing "
            "was added. The visitor needs to clear some before adding more."
        )

    try:
        page = await fetch_page(url)
    except (UnsafeUrlError, UnsupportedDocumentError) as refused:
        log_step(session_id, "add-url", f"refused — {refused}")
        return f"Could not read {url}: {refused}"
    except httpx.HTTPError as unreachable:
        logger.warning("Could not reach %s: %s", url, unreachable)
        return f"Could not reach {url}, so nothing was added."

    held = set(context.deps.documents)
    page_already_indexed = url in held
    links = [
        link
        for link in (
            extract_links(page.body, page.url)
            if "html" in page.content_type
            else []
        )
        if link.url not in held
    ]
    described = page.title or url
    if not links:
        summary = (
            f"{described} is already indexed, and every page it links to is too."
            if page_already_indexed
            else f"Index {described}? It links to no other pages on the same site."
        )
    elif page_already_indexed:
        summary = (
            f"{described} is already indexed. It links to {len(links)} page(s) "
            "that are not — choose any that should be added."
        )
    else:
        summary = (
            f"Index {described}, and choose any of the {len(links)} pages it "
            "links to that should be indexed with it."
        )
    log_step(
        session_id,
        "defer",
        f"offering {len(links)} new link(s) from {url} "
        f"(page already indexed={page_already_indexed}, {slots_left} slot(s) left)",
    )

    # A deferred call never re-enters this body — the result handed back from
    # outside *is* its return value — so the indexing happens in the resume
    # path. Anything done here would happen whether or not the visitor answers.
    raise CallDeferred(
        metadata={
            "action": "add_url_to_knowledge_base",
            "url": url,
            "page_title": page.title,
            "page_already_indexed": page_already_indexed,
            "links": [{"url": link.url, "text": link.text} for link in links],
            "slots_left": slots_left,
            "summary": summary,
        }
    )
