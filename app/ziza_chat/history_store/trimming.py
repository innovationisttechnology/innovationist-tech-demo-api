from math import ceil
from typing import Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from app.ziza_chat.history_store.summary import is_summary_turn

MAX_PROMPT_TURNS = 10
TRIM_STEP_TURNS = 5


def turns_to_drop(total_turns: int) -> int:
    if total_turns <= MAX_PROMPT_TURNS:
        return 0
    excess_turns = total_turns - MAX_PROMPT_TURNS
    return min(
        ceil(excess_turns / TRIM_STEP_TURNS) * TRIM_STEP_TURNS,
        total_turns - 1,
    )


def turns_to_summarise(stored_turns: int) -> int:
    """How many stored turns the next reply will not see.

    The message being answered is a turn too, so a conversation sitting exactly
    on the limit is already over it by the time the model is asked. Budgeting
    for it here is what keeps the summary covering everything the prompt drops:
    the loader and the summariser work from this number, and trim_to_recent_turns
    finds nothing left to cut.
    """
    incoming_turn = 1
    return turns_to_drop(stored_turns + incoming_turn)


def turn_start_indexes(messages: Sequence[ModelMessage]) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]


def split_leading_summary(
    messages: list[ModelMessage],
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    if messages and is_summary_turn(messages[0]):
        return messages[:2], messages[2:]
    return [], messages


def trim_to_recent_turns(messages: list[ModelMessage]) -> list[ModelMessage]:
    summary_turn, conversation = split_leading_summary(messages)
    turn_starts = turn_start_indexes(conversation)
    dropped_turns = turns_to_drop(len(turn_starts))
    if not dropped_turns:
        return messages
    return summary_turn + conversation[turn_starts[dropped_turns] :]
