import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from app.ziza_chat.agents.classifier import get_classifier_agent
from app.ziza_chat.agents.outputs import ClassifyResult, Intent
from app.ziza_chat.service import classify


def test_agent_is_cached() -> None:
    assert get_classifier_agent() is get_classifier_agent()


@pytest.mark.anyio
async def test_returns_valid_classification() -> None:
    """TestModel synthesizes schema-valid output, proving the agent's
    output_type wiring: whatever comes back parses into ClassifyResult."""
    agent = get_classifier_agent()
    with agent.override(model=TestModel()):
        result = await agent.run("who is Sarah?")

    classification = result.output
    assert isinstance(classification, ClassifyResult)
    assert len(classification.intents) >= 1


@pytest.mark.anyio
async def test_multi_intent_output_and_primary() -> None:
    """Force a specific model response and check it parses field-for-field,
    including enum coercion from raw strings and intent ordering."""
    forced_output = TestModel(
        custom_output_args={
            "intents": ["question", "greeting"],
            "needs_rag": True,
            "rag_query": "Sarah",
            "rag_ambiguous": True,
        }
    )
    agent = get_classifier_agent()
    with agent.override(model=forced_output):
        result = await agent.run("Hey, who is Sarah?")

    classification = result.output
    assert classification.intents == [Intent.QUESTION, Intent.GREETING]
    assert classification.primary is Intent.QUESTION
    assert classification.needs_rag is True
    assert classification.rag_query == "Sarah"
    assert classification.rag_ambiguous is True


@pytest.mark.anyio
async def test_pure_social_message_shape() -> None:
    forced_output = TestModel(
        custom_output_args={"intents": ["greeting"], "needs_rag": False}
    )
    agent = get_classifier_agent()
    with agent.override(model=forced_output):
        result = await agent.run("Hi there!")

    classification = result.output
    assert classification.primary is Intent.GREETING
    assert classification.needs_rag is False
    assert classification.rag_query is None
    assert classification.rag_ambiguous is False


@pytest.mark.anyio
async def test_service_classify_returns_classification() -> None:
    """The service-layer wrapper unwraps the agent result to ClassifyResult."""
    with get_classifier_agent().override(model=TestModel()):
        classification = await classify("what is the onboarding process?")

    assert isinstance(classification, ClassifyResult)


def test_empty_intents_rejected() -> None:
    """The schema itself guarantees at least one intent (min_length=1), so a
    model response with an empty list triggers a validation-driven retry."""
    with pytest.raises(ValidationError):
        ClassifyResult(intents=[], needs_rag=False)


def test_unknown_intent_rejected() -> None:
    with pytest.raises(ValidationError):
        ClassifyResult.model_validate(
            {"intents": ["sales pitch"], "needs_rag": False}
        )
