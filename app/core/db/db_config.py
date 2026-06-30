from typing import Any, List, Type

from beanie import Document, init_beanie
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.content_sync.models import SyncFlag
from app.core.config import settings

_client: AsyncMongoClient[Any] | None = None


def get_document_models() -> List[Type[Document]]:
    """Register all Beanie Document models here as the project grows."""
    return [SyncFlag]


def is_db_configured() -> bool:
    return bool(settings.mongo_uri)


def get_client() -> AsyncMongoClient[Any]:
    global _client
    if _client is None:
        if not settings.mongo_uri:
            raise RuntimeError("MONGO_URI is not configured")
        _client = AsyncMongoClient(settings.mongo_uri)
    return _client


def get_db() -> AsyncDatabase[Any]:
    return get_client().get_database(settings.mongo_db_name)


async def init_db() -> None:
    document_models = get_document_models()
    assert_models(document_models)

    await init_beanie(
        database=get_db(),
        document_models=document_models,
    )


async def close_db() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def assert_models(models: List[Type[Document]]) -> None:
    for model in models:
        if not issubclass(model, Document):
            raise RuntimeError(
                f"{model.__name__} is not a Beanie Document. "
                "Please update subclass document models ----> get_document_models()"
            )
