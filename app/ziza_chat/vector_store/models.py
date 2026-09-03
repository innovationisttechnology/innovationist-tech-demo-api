from datetime import datetime

from beanie import Document
from pydantic import Field
from pymongo import IndexModel

from app.core.config import settings
from app.core.utils.time import utc_now


class KnowledgeChunk(Document):


    session_id: str
    # The uploaded file or URL this came from. `source` is finer-grained
    # ("handbook.pdf (page 3)"), so counting sources would count one PDF many
    # times; the per-session upload limit counts distinct documents.
    document: str = ""
    source: str
    text: str
    text_hash: str
    embedding: list[float]
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "knowledge_chunks"
        indexes = [
            IndexModel([("session_id", 1)], name="session_chunks"),
            IndexModel(
                [("session_id", 1), ("document", 1)],
                name="session_documents",
            ),
            IndexModel(
                [("session_id", 1), ("text_hash", 1)],
                name="session_chunk_hashes",
            ),

            IndexModel(
                [("created_at", 1)],
                name="ttl_idle_chunks",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]
