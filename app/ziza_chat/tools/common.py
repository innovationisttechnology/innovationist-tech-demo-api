"""Utility tools shared by agents: things every agent may need regardless of
its role, as opposed to integration-specific tools like web_search."""

from datetime import datetime, timezone

from pydantic_ai import RunContext

from app.ziza_chat.deps import ChatDeps
from app.ziza_chat.vector_store.store import RetrievedChunk


def current_datetime() -> str:
    """Get the current date and time (UTC).

    Use this whenever an answer depends on today's date or the current time —
    e.g. "what day is it", relative dates ("next Friday"), or judging how
    recent something is. Never guess the date from prior knowledge.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%A, %B %d, %Y at %H:%M:%S UTC")


def format_retrieved_chunks(query: str, retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return f"No knowledge base passages matched {query!r}."
    return "\n\n".join(
        f"[source: {chunk.source} | relevance: {chunk.score:.2f}]\n{chunk.text}"
        for chunk in retrieved
    )


async def search_knowledge_base(context: RunContext[ChatDeps], query: str) -> str:
    """Search this session's knowledge base for passages relevant to the query.

    Call this whenever the answer might depend on a document the visitor added.
    The knowledge base holds only their own material, scoped to this session —
    it is not company-internal or public knowledge, so don't reach for it on
    general questions.

    Pass the topic or entity to look up rather than the user's whole message.
    Results come back most-relevant first, each tagged with its source label and
    a relevance score; name the source when you use a passage. Returns a plain
    "no passages matched" message when nothing is relevant — that is a real
    answer, not a failure to retry.
    """
    store = context.deps.vector_store
    if store is None:
        return "The knowledge base is not available in this session."
    retrieved = await store.search(context.deps.session_id, query)
    return format_retrieved_chunks(query, retrieved)
