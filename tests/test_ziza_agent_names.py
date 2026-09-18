"""Every agent carries an explicit name.

Without one, pydantic-ai infers the name from the variable it was assigned to.
These are all built inside `@lru_cache` factories and returned directly, so
inference finds nothing and every run span would be labelled `agent` — nine
agents indistinguishable from each other the moment Logfire is switched on.
"""

from app.ziza_chat.agents.chat import get_chat_agent
from app.ziza_chat.agents.classifier import get_classifier_agent
from app.ziza_chat.agents.conversation_summary import get_conversation_summary_agent
from app.ziza_chat.agents.followups import get_followup_grader, get_followup_writer
from app.ziza_chat.agents.page_summary import get_page_summary_agent
from app.ziza_chat.agents.starter_questions import (
    get_starter_grader,
    get_starter_question_agent,
)
from app.ziza_chat.agents.vision import get_vision_agent


def names_by_factory() -> dict[str, str | None]:
    return {
        "get_chat_agent": get_chat_agent().name,
        "get_classifier_agent": get_classifier_agent().name,
        "get_conversation_summary_agent": get_conversation_summary_agent().name,
        "get_followup_grader": get_followup_grader().name,
        "get_followup_writer": get_followup_writer().name,
        "get_page_summary_agent": get_page_summary_agent().name,
        "get_starter_grader": get_starter_grader().name,
        "get_starter_question_agent": get_starter_question_agent().name,
        "get_vision_agent": get_vision_agent().name,
    }


class TestEveryAgentIsNamed:
    def test_none_of_them_fall_back_to_the_inferred_name(self) -> None:
        named = names_by_factory()
        assert [factory for factory, name in named.items() if not name] == []

    def test_the_names_are_distinct(self) -> None:
        names = list(names_by_factory().values())
        assert len(set(names)) == len(names)
