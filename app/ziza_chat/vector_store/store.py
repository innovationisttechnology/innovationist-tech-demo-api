
import asyncio
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Any, Mapping, Protocol, Sequence

from beanie import PydanticObjectId
from beanie.operators import In
from pydantic import BaseModel, Field
from pymongo.errors import OperationFailure

from app.ziza_chat.vector_store.chunking import chunk_text, enforce_token_limit
from app.ziza_chat.vector_store.embeddings import (
    MAX_EMBED_TOKENS,
    Embedder,
    get_embedder,
)
from app.ziza_chat.vector_store.index import VECTOR_INDEX_NAME
from app.ziza_chat.vector_store.models import KnowledgeChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float


@dataclass(frozen=True)
class DocumentSection:
    text: str
    source: str
    document: str = ""


class ChunkLike(Protocol):
    text: str
    source: str
    embedding: list[float]


MIN_SCORE = 0.5

CHUNK_OVERLAP_CHARS = 150


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def unique_by_hash(chunks: list[str]) -> dict[str, str]:
    unique_chunks: dict[str, str] = {}
    for chunk in chunks:
        unique_chunks.setdefault(hash_text(chunk), chunk)
    return unique_chunks


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(
        component_a * component_b
        for component_a, component_b in zip(vector_a, vector_b, strict=True)
    )
    norm_a = sqrt(sum(component * component for component in vector_a))
    norm_b = sqrt(sum(component * component for component in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def rank_chunks(
    query_vector: list[float],
    chunks: Sequence[ChunkLike],
    limit: int,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    scored = [
        RetrievedChunk(
            text=chunk.text,
            source=chunk.source,
            score=cosine_similarity(query_vector, chunk.embedding),
        )
        for chunk in chunks
    ]
    relevant = [retrieved for retrieved in scored if retrieved.score >= min_score]
    relevant.sort(key=lambda retrieved: retrieved.score, reverse=True)
    return relevant[:limit]


class _ChunkHashView(BaseModel):
    text_hash: str


class _ChunkIdView(BaseModel):
    id: PydanticObjectId = Field(alias="_id")


class MongoVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def add_document(self, session_id: str, text: str, source: str) -> int:
        return await self.add_sections(
            session_id, [DocumentSection(text=text, source=source, document=source)]
        )

    async def count_documents(self, session_id: str) -> int:
        return len(await self.list_documents(session_id))

    async def list_documents(self, session_id: str) -> list[str]:
        collection = KnowledgeChunk.get_pymongo_collection()
        names = await collection.distinct("document", {"session_id": session_id})
        return sorted(str(name) for name in names if name)

    async def add_sections(
        self, session_id: str, sections: Sequence[DocumentSection]
    ) -> int:
        new_chunks = await asyncio.to_thread(self._prepare_chunks, sections)
        if new_chunks:
            existing_hashes = await self._existing_hashes(session_id, list(new_chunks))
            for chunk_hash in existing_hashes:
                new_chunks.pop(chunk_hash, None)
        if not new_chunks:
            return 0
        embeddings = await asyncio.to_thread(
            self._embedder.embed_passages,
            [chunk.text for chunk in new_chunks.values()],
        )
        documents = [
            KnowledgeChunk(
                session_id=session_id,
                document=chunk.document or chunk.source,
                source=chunk.source,
                text=chunk.text,
                text_hash=chunk_hash,
                embedding=embedding,
            )
            for (chunk_hash, chunk), embedding in zip(
                new_chunks.items(), embeddings, strict=True
            )
        ]
        await KnowledgeChunk.insert_many(documents)
        return len(documents)

    def _prepare_chunks(
        self, sections: Sequence[DocumentSection]
    ) -> dict[str, DocumentSection]:
        unique_chunks: dict[str, DocumentSection] = {}
        for section in sections:
            chunks = enforce_token_limit(
                chunk_text(section.text),
                self._embedder.count_tokens,
                MAX_EMBED_TOKENS,
                overlap_chars=CHUNK_OVERLAP_CHARS,
            )
            for chunk in chunks:
                unique_chunks.setdefault(
                    hash_text(chunk), DocumentSection(text=chunk, source=section.source)
                )
        return unique_chunks

    async def wait_until_searchable(
        self, session_id: str, timeout_seconds: float = 10.0
    ) -> bool:
        """Block until every stored chunk for the session is visible to
        $vectorSearch (mongot indexes new documents asynchronously).

        Returns False on timeout; True immediately when there is nothing to
        index or the deployment has no Atlas search (the Python fallback is
        always consistent with the collection).
        """
        sample_chunk = await KnowledgeChunk.find_one(
            KnowledgeChunk.session_id == session_id
        )
        if sample_chunk is None:
            return True
        id_views = (
            await KnowledgeChunk.find(KnowledgeChunk.session_id == session_id)
            .project(_ChunkIdView)
            .to_list()
        )
        expected_ids = {view.id for view in id_views}
        pipeline: list[Mapping[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": sample_chunk.embedding,
                    "exact": True,
                    "limit": len(expected_ids),
                    "filter": {"session_id": session_id},
                }
            },
            {"$project": {"_id": 1}},
        ]
        poll_seconds = 0.25
        attempts = max(1, int(timeout_seconds / poll_seconds))
        for _ in range(attempts):
            try:
                results = await KnowledgeChunk.aggregate(pipeline).to_list()
            except OperationFailure:
                return True
            indexed_ids = {result["_id"] for result in results}
            if expected_ids <= indexed_ids:
                return True
            await asyncio.sleep(poll_seconds)
        logger.warning(
            "Chunks for session %s not searchable after %.1fs",
            session_id,
            timeout_seconds,
        )
        return False

    async def _existing_hashes(self, session_id: str, hashes: list[str]) -> set[str]:
        views = (
            await KnowledgeChunk.find(
                KnowledgeChunk.session_id == session_id,
                In(KnowledgeChunk.text_hash, hashes),
            )
            .project(_ChunkHashView)
            .to_list()
        )
        return {view.text_hash for view in views}

    async def search(
        self,
        session_id: str,
        query: str,
        limit: int = 4,
        min_score: float = MIN_SCORE,
    ) -> list[RetrievedChunk]:
        query_vector = await asyncio.to_thread(self._embedder.embed_query, query)
        try:
            return await self._search_atlas(session_id, query_vector, limit, min_score)
        except OperationFailure as failure:
            logger.warning(
                "$vectorSearch unavailable (%s) — ranking in Python instead",
                failure.details.get("errmsg") if failure.details else failure,
            )
            return await self._search_python(session_id, query_vector, limit, min_score)

    async def _search_atlas(
        self,
        session_id: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        pipeline: list[Mapping[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(limit * 10, 100),
                    "limit": limit,
                    "filter": {"session_id": session_id},
                }
            },
            {
                "$project": {
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results = await KnowledgeChunk.aggregate(pipeline).to_list()
        retrieved: list[RetrievedChunk] = []
        for result in results:
            # Atlas reports cosine normalized to (1 + cosine) / 2; undo it so
            # scores stay comparable with the Python path and MIN_SCORE means
            # one thing everywhere.
            raw_cosine = 2.0 * float(result["score"]) - 1.0
            if raw_cosine < min_score:
                continue
            retrieved.append(
                RetrievedChunk(
                    text=result["text"], source=result["source"], score=raw_cosine
                )
            )
        return retrieved

    async def _search_python(
        self,
        session_id: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        chunks = await KnowledgeChunk.find(
            KnowledgeChunk.session_id == session_id
        ).to_list()
        return rank_chunks(query_vector, chunks, limit, min_score)

    async def clear(self, session_id: str) -> int:
        result = await KnowledgeChunk.find(
            KnowledgeChunk.session_id == session_id
        ).delete()
        return int(result.deleted_count) if result else 0


@lru_cache
def get_vector_store() -> MongoVectorStore:
    return MongoVectorStore(get_embedder())
