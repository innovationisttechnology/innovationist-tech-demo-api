"""Tests for reading a conversation back.

The stored format is what gets replayed to the model, so it carries more than a
visitor saw. This is about what comes out of it.
"""

from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from app.ziza_chat import service
from app.ziza_chat.history_store import service as history_service
from app.ziza_chat.history_store.summary import build_summary_turn
from app.ziza_chat.history_store.transcript import TranscriptTurn, to_transcript

AT = datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)


def answered_turn() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content="who is Sarah?")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="search_knowledge_base",
                    args={"query": "Sarah"},
                    tool_call_id="call_1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="search_knowledge_base",
                    content="[source: handbook.pdf] Sarah leads the platform team.",
                    tool_call_id="call_1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="Sarah leads the platform team.")]),
    ]


class TestWhatComesOut:
    def test_the_question_and_the_answer(self) -> None:
        turns = to_transcript(answered_turn(), AT)
        assert [(t.role, t.text) for t in turns] == [
            ("user", "who is Sarah?"),
            ("assistant", "Sarah leads the platform team."),
        ]

    def test_tool_calls_and_passages_are_left_out(self) -> None:
        """How the answer was reached, not part of it — and the panel that
        shows those is fed from the live stream."""
        text = " ".join(turn.text for turn in to_transcript(answered_turn(), AT))
        assert "search_knowledge_base" not in text
        assert "[source:" not in text

    def test_the_rolling_summary_is_left_out(self) -> None:
        """It is written for the model and reads as something the visitor said."""
        messages = build_summary_turn("They asked about releases.") + answered_turn()
        turns = to_transcript(messages, AT)
        assert [t.role for t in turns] == ["user", "assistant"]
        assert all("Summary of earlier" not in t.text for t in turns)

    def test_an_empty_answer_is_not_a_turn(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="hi")]),
            ModelResponse(parts=[TextPart(content="   ")]),
        ]
        assert [t.role for t in to_transcript(messages, AT)] == ["user"]

    def test_a_refusal_is_a_real_turn(self) -> None:
        """The gate's answers are stored so the transcript has no holes."""
        from app.ziza_chat.history_store.store import build_refusal_turn

        turns = to_transcript(
            build_refusal_turn("what is gravity?", "I can only answer from..."), AT
        )
        assert [t.role for t in turns] == ["user", "assistant"]


class TestTheResponse:
    @pytest.mark.anyio
    async def test_turns_come_back_in_order_with_stable_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def load(session_id: str) -> list[TranscriptTurn]:
            return [
                TranscriptTurn(role="user", text="first", at=AT),
                TranscriptTurn(role="assistant", text="second", at=AT),
            ]

        monkeypatch.setattr(service, "load_transcript", load)
        response = await service.chat_history("s")
        assert [(t.id, t.role, t.text) for t in response.turns] == [
            ("0", "user", "first"),
            ("1", "assistant", "second"),
        ]

    @pytest.mark.anyio
    async def test_no_database_is_an_empty_conversation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(history_service, "is_db_configured", lambda: False)
        response = await service.chat_history("s")
        assert response.turns == []
