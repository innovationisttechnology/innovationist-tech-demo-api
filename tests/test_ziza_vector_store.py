"""Unit tests for the ziza_chat vector store.

Everything here runs without MongoDB and without downloading the embedding
model: chunking and ranking are pure functions, and the knowledge-base tool
is exercised with a stub store.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import RunContext

from app.ziza_chat.deps import ChatDeps
from app.ziza_chat.tools.common import format_retrieved_chunks, search_knowledge_base
from app.ziza_chat.vector_store.chunking import chunk_text
from app.ziza_chat.vector_store.store import (
    ChunkLike,
    DocumentSection,
    MongoVectorStore,
    RetrievedChunk,
    cosine_similarity,
    hash_text,
    rank_chunks,
    unique_by_hash,
)


def make_chunk(text: str, embedding: list[float]) -> ChunkLike:
    return cast(
        ChunkLike,
        SimpleNamespace(text=text, source="handbook", embedding=embedding),
    )


class TestChunkText:
    def test_short_text_is_a_single_chunk(self) -> None:
        assert chunk_text("Hello world.") == ["Hello world."]

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("   \n\n  ") == []

    def test_paragraphs_are_packed_together_while_they_fit(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert chunk_text(text, max_chars=100) == [text]

    def test_paragraphs_split_when_over_the_limit(self) -> None:
        first_paragraph = "alpha " * 10
        second_paragraph = "bravo " * 10
        chunks = chunk_text(
            f"{first_paragraph.strip()}\n\n{second_paragraph.strip()}", max_chars=70
        )
        assert len(chunks) == 2
        assert "alpha" in chunks[0] and "bravo" in chunks[1]

    def test_oversized_paragraph_is_hard_split_with_overlap(self) -> None:
        long_paragraph = "x" * 250
        chunks = chunk_text(long_paragraph, max_chars=100, overlap_chars=20)
        assert all(len(chunk) <= 100 for chunk in chunks)
        assert sum(len(chunk) for chunk in chunks) > 250

    def test_rejects_overlap_larger_than_max(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("anything", max_chars=100, overlap_chars=100)


class TestRanking:
    def test_cosine_similarity_of_identical_vectors_is_one(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_cosine_similarity_of_orthogonal_vectors_is_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_scores_zero_instead_of_dividing_by_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_rank_chunks_orders_by_similarity_and_limits(self) -> None:
        query_vector = [1.0, 0.0]
        chunks = [
            make_chunk("orthogonal", [0.0, 1.0]),
            make_chunk("exact", [1.0, 0.0]),
            make_chunk("close", [1.0, 0.5]),
        ]
        ranked = rank_chunks(query_vector, chunks, limit=2)
        assert [retrieved.text for retrieved in ranked] == ["exact", "close"]
        assert ranked[0].score == pytest.approx(1.0)

    def test_rank_chunks_drops_scores_below_min_score(self) -> None:
        query_vector = [1.0, 0.0]
        chunks = [
            make_chunk("exact", [1.0, 0.0]),
            make_chunk("orthogonal", [0.0, 1.0]),
        ]
        ranked = rank_chunks(query_vector, chunks, limit=5, min_score=0.5)
        assert [retrieved.text for retrieved in ranked] == ["exact"]


class TestDedup:
    def test_hash_text_is_deterministic_and_content_sensitive(self) -> None:
        assert hash_text("same") == hash_text("same")
        assert hash_text("same") != hash_text("different")

    def test_unique_by_hash_drops_repeats_and_keeps_order(self) -> None:
        deduped = unique_by_hash(["alpha", "bravo", "alpha", "charlie", "bravo"])
        assert list(deduped.values()) == ["alpha", "bravo", "charlie"]
        assert set(deduped.keys()) == {
            hash_text("alpha"),
            hash_text("bravo"),
            hash_text("charlie"),
        }


class TestSearchKnowledgeBaseTool:
    @pytest.mark.anyio
    async def test_reports_unavailable_without_a_store(self) -> None:
        deps = ChatDeps(session_id="session-1", vector_store=None)
        context = cast(RunContext[ChatDeps], SimpleNamespace(deps=deps))
        result = await search_knowledge_base(context, "who is Sarah?")
        assert "not available" in result

    @pytest.mark.anyio
    async def test_formats_results_from_the_store(self) -> None:
        class StubStore:
            async def search(
                self, session_id: str, query: str, limit: int = 4
            ) -> list[RetrievedChunk]:
                assert session_id == "session-1"
                return [
                    RetrievedChunk(
                        text="Sarah leads the platform team.",
                        source="handbook",
                        score=0.91,
                    )
                ]

        deps = ChatDeps(
            session_id="session-1", vector_store=cast(MongoVectorStore, StubStore())
        )
        context = cast(RunContext[ChatDeps], SimpleNamespace(deps=deps))
        result = await search_knowledge_base(context, "who is Sarah?")
        assert "Sarah leads the platform team." in result
        assert "handbook" in result
        assert "0.91" in result

    def test_format_reports_no_matches(self) -> None:
        assert "No knowledge base passages" in format_retrieved_chunks("query", [])


class FakeEmbedder:
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


def prepare(sections: list[DocumentSection]) -> list[DocumentSection]:
    store = MongoVectorStore(FakeEmbedder())
    return list(store._prepare_chunks(sections).values())


def stored_document_names(chunks: list[DocumentSection]) -> set[str]:
    return {chunk.document or chunk.source for chunk in chunks}


class TestPrepareChunks:
    def test_every_chunk_keeps_the_document_it_came_from(self) -> None:
        pages = [
            DocumentSection(
                text=f"Body text of page {page}. " * 8,
                source=f"handbook.pdf (page {page})",
                document="handbook.pdf",
            )
            for page in range(1, 13)
        ]
        assert stored_document_names(prepare(pages)) == {"handbook.pdf"}

    def test_sources_stay_page_level_for_citations(self) -> None:
        pages = [
            DocumentSection(
                text=f"Body text of page {page}. " * 8,
                source=f"handbook.pdf (page {page})",
                document="handbook.pdf",
            )
            for page in range(1, 4)
        ]
        assert {chunk.source for chunk in prepare(pages)} == {
            "handbook.pdf (page 1)",
            "handbook.pdf (page 2)",
            "handbook.pdf (page 3)",
        }

    def test_a_url_and_its_summary_are_one_document(self) -> None:
        url = "https://example.com/guide"
        sections = [
            DocumentSection(text="Page body. " * 20, source=f"Guide ({url})", document=url),
            DocumentSection(
                text="Overview of the guide. " * 20,
                source=f"Guide ({url}) [summary]",
                document=url,
            ),
        ]
        assert stored_document_names(prepare(sections)) == {url}

    def test_falls_back_to_the_source_when_no_document_is_given(self) -> None:
        section = DocumentSection(text="Pasted text. " * 20, source="pasted-notes")
        assert stored_document_names(prepare([section])) == {"pasted-notes"}
