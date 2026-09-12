import logging

from app.ziza_chat.caption_cache import clear_captions
from app.ziza_chat.history_store.service import clear_history
from app.ziza_chat.page_links import clear_page_links
from app.ziza_chat.vector_store.store import get_vector_store

logger = logging.getLogger(__name__)


async def clear_knowledge(session_id: str) -> int:
    # Captions are derived from the visitor's images, so clearing the knowledge
    # base has to take them too or their content survives the delete.
    captions_removed = await clear_captions(session_id)
    if captions_removed:
        logger.info("cleared %d cached caption(s) for %s", captions_removed, session_id)
    await clear_page_links(session_id)
    await clear_history(session_id)
    return await get_vector_store().clear(session_id)


async def describe_holdings(session_id: str) -> list[str]:
    return await get_vector_store().list_documents(session_id)
