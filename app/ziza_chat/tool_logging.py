import logging
from typing import AsyncIterable

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
)

from app.ziza_chat.deps import ChatDeps

logger = logging.getLogger(__name__)

MAX_LOGGED_CHARS = 160

SESSION_PREFIX_CHARS = 8


def summarize(value: object) -> str:
    text = " ".join(str(value).split())
    if len(text) <= MAX_LOGGED_CHARS:
        return text
    return f"{text[:MAX_LOGGED_CHARS]}… (+{len(text) - MAX_LOGGED_CHARS} chars)"


def log_step(session_id: str, step: str, detail: str) -> None:
    logger.info(
        "[%s] %-9s %s", session_id[:SESSION_PREFIX_CHARS], step, summarize(detail)
    )


async def log_tool_events(
    context: RunContext[ChatDeps], events: AsyncIterable[AgentStreamEvent]
) -> None:
    session_id = context.deps.session_id
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            log_step(
                session_id,
                "tool→",
                f"{event.part.tool_name}({summarize(event.part.args)})",
            )
        elif isinstance(event, FunctionToolResultEvent):
            outcome = "retry" if isinstance(event.part, RetryPromptPart) else "←tool"
            log_step(
                session_id,
                outcome,
                f"{event.part.tool_name} -> {summarize(event.part.content)}",
            )
