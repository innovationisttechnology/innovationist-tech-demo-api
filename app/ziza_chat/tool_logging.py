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


def summarize(value: object) -> str:
    text = " ".join(str(value).split())
    if len(text) <= MAX_LOGGED_CHARS:
        return text
    return f"{text[:MAX_LOGGED_CHARS]}… (+{len(text) - MAX_LOGGED_CHARS} chars)"


async def log_tool_events(
    context: RunContext[ChatDeps], events: AsyncIterable[AgentStreamEvent]
) -> None:
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            logger.info(
                "tool call   %s(%s)",
                event.part.tool_name,
                summarize(event.part.args),
            )
        elif isinstance(event, FunctionToolResultEvent):
            outcome = "retry" if isinstance(event.part, RetryPromptPart) else "return"
            logger.info(
                "tool %-6s %s -> %s",
                outcome,
                event.part.tool_name,
                summarize(event.part.content),
            )
