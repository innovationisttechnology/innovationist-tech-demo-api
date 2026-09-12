from dataclasses import dataclass, field

from app.ziza_chat.vector_store.store import MongoVectorStore


@dataclass
class SearchOutcome:
    query: str
    matches: int
    best_score: float | None


@dataclass
class ChatDeps:
    """Per-request dependencies handed to the chat agent via RunContext.

    Extend this with clients the tools need (HTTP client, etc.) as the module
    grows — it is the single place tools reach for shared state.
    """

    session_id: str
    intent: str = "unknown"
    # Names of the documents this session holds, so the agent can say what it
    # has without searching — otherwise it guesses, and wrongly claims an empty
    # knowledge base whenever a query happens not to match.
    documents: list[str] = field(default_factory=list)
    # None when MongoDB is not configured; the search tool degrades gracefully.
    vector_store: MongoVectorStore | None = None
    # A page this message pointed at that the session does not hold yet, so
    # the agent is told which URL to pass rather than having to find it.
    mentioned_url: str | None = None
    # Written by search_knowledge_base and read after the run, so the caller
    # knows how retrieval actually went without parsing the answer text.
    searches: list[SearchOutcome] = field(default_factory=list)
