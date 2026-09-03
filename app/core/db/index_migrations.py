"""Index changes MongoDB will not make for us.

`create_index` cannot alter an existing index: changing a TTL raises
IndexOptionsConflict rather than updating it, which fails `init_beanie` and so
fails startup. Beanie also never drops an index it no longer declares. Both are
reconciled here, from the models' own declarations, before Beanie initialises.
"""

import logging
from typing import Any, List, Type

from beanie import Document
from pymongo import IndexModel
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

# Indexes an earlier version of a model declared, keyed by collection. Beanie
# leaves these in place, and a stale unique index keeps rejecting writes the
# current schema allows.
SUPERSEDED_INDEXES: dict[str, tuple[str, ...]] = {
    # Captions were keyed on image_hash alone before they were session-scoped.
    "image_captions": ("uniq_image_hash",),
}


def collection_name(model: Type[Document]) -> str:
    return str(getattr(model.Settings, "name", model.__name__.lower()))


def declared_ttl_indexes(model: Type[Document]) -> dict[str, int]:
    """TTL indexes the model declares, as {index name: expireAfterSeconds}."""
    declared: dict[str, int] = {}
    for index in getattr(model.Settings, "indexes", []) or []:
        if not isinstance(index, IndexModel):
            continue
        specification: dict[str, Any] = index.document
        name = specification.get("name")
        ttl_seconds = specification.get("expireAfterSeconds")
        if name and ttl_seconds is not None:
            declared[str(name)] = int(ttl_seconds)
    return declared


async def reconcile_indexes(
    database: AsyncDatabase[Any], models: List[Type[Document]]
) -> None:
    for model in models:
        name = collection_name(model)
        collection = database[name]
        try:
            live = await collection.index_information()
        except PyMongoError as failure:
            logger.warning("Could not inspect indexes on %s: %s", name, failure)
            continue

        for index_name in SUPERSEDED_INDEXES.get(name, ()):
            if index_name in live:
                await collection.drop_index(index_name)
                logger.info("Dropped superseded index %r on %s", index_name, name)

        for index_name, ttl_seconds in declared_ttl_indexes(model).items():
            existing = live.get(index_name)
            if existing is None:
                continue
            current = existing.get("expireAfterSeconds")
            if current != ttl_seconds:
                await collection.drop_index(index_name)
                logger.info(
                    "Dropped TTL index %r on %s (was %ss, now %ss) so Beanie "
                    "recreates it",
                    index_name,
                    name,
                    current,
                    ttl_seconds,
                )
