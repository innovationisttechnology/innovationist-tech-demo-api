from pydantic_ai import RunContext

from app.ziza_chat.config import ziza_settings
from app.ziza_chat.deps import ChatDeps


async def web_search(context: RunContext[ChatDeps], query: str) -> str:
    """Search the web for up-to-date information via Tavily.

    Stub: returns a placeholder until TAVILY_API_KEY is set and a client is
    wired in. `context` is accepted so this matches the pydantic-ai tool
    signature and can read per-request deps later.
    """
    if not ziza_settings.tavily_api_key:
        return f"[web search disabled — set TAVILY_API_KEY. query was {query!r}]"
    return f"[web search not implemented yet — query was {query!r}]"
