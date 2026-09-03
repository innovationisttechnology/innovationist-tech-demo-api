import logging

from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from app.ziza_chat.config import ziza_settings
from app.ziza_chat.vector_store.embeddings import get_embedding_dimensions
from app.ziza_chat.vector_store.models import KnowledgeChunk

logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "knowledge_vector_index"


async def ensure_vector_index() -> bool:
    """Create the $vectorSearch index if it does not exist yet.

    Returns False when the deployment has no search support (plain MongoDB
    instead of Atlas / atlas-local) — search then falls back to ranking in
    Python instead of failing.
    """
    collection = KnowledgeChunk.get_pymongo_collection()

    try:
        existing = [
            index["name"] async for index in await collection.list_search_indexes()
        ]
    except OperationFailure:
        logger.warning(
            "Search indexes unavailable (not an Atlas deployment?) — "
            "knowledge base search will rank in Python instead."
        )
        return False

    if VECTOR_INDEX_NAME in existing:
        logger.info("Vector index %r already exists", VECTOR_INDEX_NAME)
        return True

    dimensions = get_embedding_dimensions(ziza_settings.embedding_model_id)
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "session_id"},
            ]
        },
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )
    await collection.create_search_index(model)
    logger.info(
        "Vector index %r created (%d dims, cosine)", VECTOR_INDEX_NAME, dimensions
    )
    return True
