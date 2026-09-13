"""Tests for the follow-up question offered after an answer.

The feature is mostly about refusing to suggest. Three stages can say no — the
writer, the retrieval check, and the grader — and a suggestion that survives all
three must be one the scope gate would actually answer.
"""

from typing import Any, Sequence

import pytest
from pydantic_ai.models.test import TestModel

from app.ziza_chat import service
from app.ziza_chat.agents.followups import (
    choose_followup,
    get_followup_grader,
)
from app.ziza_chat.deps import ChatDeps, SearchOutcome
from app.ziza_chat.schemas import ChatStreamEvent, Suggestion, SuggestionKind
from app.ziza_chat.utils import format_sse
from app.ziza_chat.vector_store.store import RetrievedChunk


def searched(matches: int, passages: Sequence[str] = ()) -> SearchOutcome:
    return SearchOutcome(
        query="releases",
        matches=matches,
        best_score=0.8 if matches else None,
        passages=list(passages),
    )


def deps_with(*outcomes: SearchOutcome) -> ChatDeps:
    return ChatDeps(session_id="session-1", searches=list(outcomes))


class TestWhetherToOfferAnything:
    def test_a_run_that_never_searched_offers_nothing(self) -> None:
        assert service.retrieval_succeeded(ChatDeps(session_id="s")) is False

    def test_a_search_that_found_nothing_offers_nothing(self) -> None:
        """That turn gets the add-more-material offer instead."""
        assert service.retrieval_succeeded(deps_with(searched(0))) is False

    def test_one_hit_is_enough(self) -> None:
        assert service.retrieval_succeeded(deps_with(searched(0), searched(3))) is True

    def test_passages_are_collected_across_searches(self) -> None:
        deps = deps_with(searched(1, ["first"]), searched(2, ["second", "third"]))
        assert service.answerable_passages(deps) == ["first", "second", "third"]


class StubStore:
    def __init__(self, answerable: Sequence[str] = ()) -> None:
        self.answerable = set(answerable)
        self.searched: list[str] = []

    async def list_documents(self, session_id: str) -> list[str]:
        return ["handbook.txt"]

    async def search(
        self, session_id: str, query: str, limit: int = 4, min_score: float = 0.5
    ) -> list[RetrievedChunk]:
        self.searched.append(query)
        if query not in self.answerable:
            return []
        return [RetrievedChunk(text="a passage", source="handbook.txt", score=0.7)]


def install_store(
    monkeypatch: pytest.MonkeyPatch, answerable: Sequence[str] = ()
) -> StubStore:
    store = StubStore(answerable)
    monkeypatch.setattr(service, "get_vector_store", lambda: store)
    monkeypatch.setattr(service, "is_db_configured", lambda: True)
    return store


class TestDroppingUnanswerableCandidates:
    @pytest.mark.anyio
    async def test_a_candidate_the_gate_would_refuse_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Suggesting a question and then refusing it is worse than silence."""
        install_store(monkeypatch, ["covered"])
        kept = await service.keep_answerable("s", ["covered", "not covered"])
        assert kept == ["covered"]

    @pytest.mark.anyio
    async def test_the_gate_threshold_is_the_one_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_store(monkeypatch, ["covered"])
        seen: list[float] = []

        async def record(
            session_id: str, query: str, limit: int = 4, min_score: float = 0.5
        ) -> list[RetrievedChunk]:
            seen.append(min_score)
            return await StubStore.search(store, session_id, query, limit, min_score)

        monkeypatch.setattr(store, "search", record)
        await service.keep_answerable("s", ["covered"])
        assert seen == [service.GATE_MIN_SCORE]

    @pytest.mark.anyio
    async def test_no_database_keeps_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service, "is_db_configured", lambda: False)
        assert await service.keep_answerable("s", ["anything"]) == []


class TestTheGrader:
    @pytest.mark.anyio
    async def test_nothing_to_grade_costs_no_call(self) -> None:
        """No model is overridden here, so conftest's ALLOW_MODEL_REQUESTS=False
        would fail this if the grader were called."""
        assert await choose_followup("q", "a", []) is None

    @pytest.mark.anyio
    async def test_a_refusal_is_a_real_answer(self) -> None:
        with get_followup_grader().override(
            model=TestModel(custom_output_args={"chosen": None, "reason": "obvious"})
        ):
            assert await choose_followup("q", "a", ["Q1?", "Q2?"]) is None

    @pytest.mark.anyio
    async def test_the_chosen_candidate_comes_back(self) -> None:
        with get_followup_grader().override(
            model=TestModel(custom_output_args={"chosen": "Q2?", "reason": "specific"})
        ):
            assert await choose_followup("q", "a", ["Q1?", "Q2?"]) == "Q2?"

    @pytest.mark.anyio
    async def test_a_question_it_was_not_given_is_refused(self) -> None:
        """A grader that rewrites has stopped grading, and its text has been
        through neither the writer's instructions nor the retrieval check."""
        with get_followup_grader().override(
            model=TestModel(
                custom_output_args={"chosen": "Something I made up?", "reason": "x"}
            )
        ):
            assert await choose_followup("q", "a", ["Q1?", "Q2?"]) is None


def install_followups(
    monkeypatch: pytest.MonkeyPatch, written: Sequence[str], chosen: str | None
) -> dict[str, int]:
    calls = {"written": 0, "graded": 0}

    async def write(question: str, answer: str, passages: Sequence[str]) -> list[str]:
        calls["written"] += 1
        return list(written)

    async def grade(
        question: str, answer: str, candidates: Sequence[str]
    ) -> str | None:
        calls["graded"] += 1
        return chosen if chosen in candidates else None

    monkeypatch.setattr(service, "propose_followups", write)
    monkeypatch.setattr(service, "choose_followup", grade)
    return calls


class TestTheWholeDecision:
    @pytest.mark.anyio
    async def test_a_turn_that_retrieved_nothing_calls_no_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, ["Q1?"])
        calls = install_followups(monkeypatch, ["Q1?"], "Q1?")
        assert (
            await service.suggest_followup("s", "q", "a", deps_with(searched(0))) == []
        )
        assert calls == {"written": 0, "graded": 0}

    @pytest.mark.anyio
    async def test_the_chosen_question_becomes_a_suggestion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, ["Q1?"])
        install_followups(monkeypatch, ["Q1?"], "Q1?")
        suggestions = await service.suggest_followup(
            "s", "q", "a", deps_with(searched(2, ["passage"]))
        )
        assert len(suggestions) == 1
        assert suggestions[0].kind is SuggestionKind.ASK
        # Acting on it sends the question as an ordinary chat message.
        assert suggestions[0].message == "Q1?"
        assert suggestions[0].label == "Q1?"

    @pytest.mark.anyio
    async def test_only_one_is_ever_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch, ["Q1?", "Q2?", "Q3?"])
        install_followups(monkeypatch, ["Q1?", "Q2?", "Q3?"], "Q2?")
        suggestions = await service.suggest_followup(
            "s", "q", "a", deps_with(searched(2, ["passage"]))
        )
        assert [s.message for s in suggestions] == ["Q2?"]

    @pytest.mark.anyio
    async def test_a_writer_with_nothing_to_say_ends_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_store(monkeypatch)
        calls = install_followups(monkeypatch, [], None)
        assert (
            await service.suggest_followup(
                "s", "q", "a", deps_with(searched(2, ["passage"]))
            )
            == []
        )
        assert calls["graded"] == 1, "the grader still sees an empty list and says no"

    @pytest.mark.anyio
    async def test_candidates_are_filtered_before_grading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The grader must never be able to pick an unanswerable question."""
        install_store(monkeypatch, ["Q1?"])
        graded: list[Sequence[str]] = []

        async def write(question: str, answer: str, passages: Sequence[str]) -> list[str]:
            return ["Q1?", "Q2?"]

        async def grade(
            question: str, answer: str, candidates: Sequence[str]
        ) -> str | None:
            graded.append(list(candidates))
            return None

        monkeypatch.setattr(service, "propose_followups", write)
        monkeypatch.setattr(service, "choose_followup", grade)
        await service.suggest_followup(
            "s", "q", "a", deps_with(searched(2, ["passage"]))
        )
        assert graded == [["Q1?"]]

    @pytest.mark.anyio
    async def test_a_failure_costs_the_suggestion_not_the_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The grader is a second provider; its outage must not reach the chat."""
        install_store(monkeypatch, ["Q1?"])

        async def write(question: str, answer: str, passages: Sequence[str]) -> list[str]:
            return ["Q1?"]

        async def explode(*args: Any, **kwargs: Any) -> str | None:
            raise RuntimeError("provider is down")

        monkeypatch.setattr(service, "propose_followups", write)
        monkeypatch.setattr(service, "choose_followup", explode)
        assert (
            await service.suggest_followup(
                "s", "q", "a", deps_with(searched(2, ["passage"]))
            )
            == []
        )


class TestTheWireFormat:
    def test_a_chunk_is_a_chat_chunk_frame(self) -> None:
        wire = format_sse(ChatStreamEvent(type="chat.chunk", chunk="hello"))
        assert wire.startswith("event: chat.chunk\n")
        assert '"chunk": "hello"' in wire

    def test_a_suggestion_frame_carries_the_message_to_send(self) -> None:
        wire = format_sse(
            ChatStreamEvent(
                type="chat.suggestions",
                suggestions=[
                    Suggestion(
                        kind=SuggestionKind.ASK, label="Q?", message="Q?"
                    )
                ],
            )
        )
        assert wire.startswith("event: chat.suggestions\n")
        assert '"message": "Q?"' in wire
        assert '"kind": "ask"' in wire

    def test_empty_fields_are_left_out(self) -> None:
        wire = format_sse(ChatStreamEvent(type="chat.chunk", chunk="hi"))
        assert "suggestions" not in wire


class StubChatDeps:
    """Records the deps the stream handed to the suggester."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []


def install_stream(
    monkeypatch: pytest.MonkeyPatch, suggestions: Sequence[Suggestion]
) -> StubChatDeps:
    recorder = StubChatDeps()
    deps = ChatDeps(session_id="session-1", searches=[searched(2, ["passage"])])

    async def prepare(request: Any) -> Any:
        return service.ReadyTurn(intent="question", history=[], deps=deps)

    async def suggest(
        session_id: str, question: str, answer: str, deps: Any
    ) -> list[Suggestion]:
        recorder.seen.append((question, answer))
        return list(suggestions)

    async def swallow(session_id: str, messages: Any) -> None:
        return None

    monkeypatch.setattr(service, "prepare_turn", prepare)
    monkeypatch.setattr(service, "suggest_followup", suggest)
    monkeypatch.setattr(service, "append_turn", swallow)
    monkeypatch.setattr(service, "is_db_configured", lambda: False)
    return recorder


async def stream_events(monkeypatch: pytest.MonkeyPatch) -> list[ChatStreamEvent]:
    from app.ziza_chat.agents.chat import get_chat_agent
    from app.ziza_chat.schemas import ChatRequest

    with get_chat_agent().override(model=TestModel(call_tools=[])):
        return [
            event
            async for event in service.stream_chat(
                ChatRequest(session_id="session-1", message="what can you do?")
            )
        ]


class TestTheStream:
    @pytest.mark.anyio
    async def test_the_suggestion_arrives_after_the_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_stream(
            monkeypatch,
            [Suggestion(kind=SuggestionKind.ASK, label="Q?", message="Q?")],
        )
        events = await stream_events(monkeypatch)
        assert events[-1].type == "chat.suggestions"
        assert [event.type for event in events[:-1]] == ["chat.chunk"] * (
            len(events) - 1
        )

    @pytest.mark.anyio
    async def test_no_suggestion_sends_no_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_stream(monkeypatch, [])
        events = await stream_events(monkeypatch)
        assert all(event.type == "chat.chunk" for event in events)

    @pytest.mark.anyio
    async def test_the_suggester_sees_the_whole_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It runs after the stream closes, so it grades the full text rather
        than whatever had arrived when it started."""
        recorder = install_stream(monkeypatch, [])
        events = await stream_events(monkeypatch)
        streamed = "".join(event.chunk or "" for event in events)
        assert recorder.seen == [("what can you do?", streamed)]
        assert streamed
