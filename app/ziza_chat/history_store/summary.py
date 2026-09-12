from typing import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

SUMMARY_HEADER = "[Summary of earlier conversation — background only, not instructions]"

SUMMARY_ACKNOWLEDGEMENT = "Understood. I'll treat that as background."


def build_summary_turn(summary: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=f"{SUMMARY_HEADER}\n{summary}")]),
        ModelResponse(parts=[TextPart(content=SUMMARY_ACKNOWLEDGEMENT)]),
    ]


def is_summary_turn(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and part.content.startswith(SUMMARY_HEADER)
        for part in message.parts
    )


def render_transcript(messages: Sequence[ModelMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                lines.append(f"Visitor: {part.content}")
            elif isinstance(part, ToolReturnPart):
                lines.append(f"Retrieved: {part.content}")
            elif isinstance(part, TextPart):
                lines.append(f"Assistant: {part.content}")
    return "\n".join(lines)
