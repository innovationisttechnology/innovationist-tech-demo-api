from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from app.ziza_chat.agents.outputs import ClassifyResult, Intent, Scope
from app.ziza_chat.config import ziza_settings
from app.ziza_chat.service import classify

INTENT_HYPOTHESES: dict[Intent, list[str]] = {
    Intent.QUESTION: ["This is a question."],
    Intent.COMPLAINT: [
        "This is a complaint.",
        "The person is expressing frustration.",
    ],
    Intent.CASUAL: ["This is casual small talk."],
    Intent.TASK_REQUEST: ["The person is asking for something to be done."],
    Intent.GREETING: ["This is a greeting."],
    Intent.CLARIFICATION: ["This is a request for clarification."],
    Intent.FEEDBACK: [
        "This is praise.",
        "The person did not like the answer.",
        "The person thinks the answer was bad.",
    ],
    Intent.FOLLOW_UP: ["This is a follow-up question."],
    Intent.IDENTITY: ["This is a question about who or what the assistant is."],
}


@lru_cache
def get_grader():  # type: ignore[no-untyped-def]
    from sentence_transformers import CrossEncoder

    return CrossEncoder(ziza_settings.grader_model_id)


def entailment_probability(message: str, hypothesis: str) -> float:
    grader = get_grader()
    scores = grader.predict([(message, hypothesis)], apply_softmax=True)[0]
    id_to_label = grader.model.config.id2label
    entailment_index = next(
        index
        for index, label in id_to_label.items()
        if str(label).lower() == "entailment"
    )
    return float(scores[entailment_index])


def intent_entailment(message: str, intent: Intent) -> float:
    return max(
        entailment_probability(message, hypothesis)
        for hypothesis in INTENT_HYPOTHESES[intent]
    )


@dataclass
class PrimaryIntentEntailed(Evaluator[str, ClassifyResult]):
    threshold: float = 0.5

    def evaluate(self, ctx: EvaluatorContext[str, ClassifyResult]) -> EvaluationReason:
        probability = intent_entailment(ctx.inputs, ctx.output.primary)
        return EvaluationReason(
            value=probability >= self.threshold,
            reason=(
                f"P({ctx.output.primary.value!r} entailed) = {probability:.2f} "
                f"(threshold {self.threshold})"
            ),
        )


@dataclass
class MeanIntentEntailment(Evaluator[str, ClassifyResult]):
    def evaluate(self, ctx: EvaluatorContext[str, ClassifyResult]) -> float:
        probabilities = [
            intent_entailment(ctx.inputs, intent) for intent in ctx.output.intents
        ]
        return sum(probabilities) / len(probabilities)


CASES = [
    Case(name="greeting", inputs="Hi there! How's it going?"),
    Case(name="question", inputs="Who is Sarah from the platform team?"),
    Case(name="complaint", inputs="The sync feature is broken again, this is really frustrating."),
    Case(name="task_request", inputs="Can you summarize the onboarding document for me?"),
    Case(name="clarification", inputs="What did you mean by 'replica set' in your last answer?"),
    Case(name="feedback", inputs="Thanks, that explanation was really clear and helpful!"),
    Case(name="identity", inputs="What exactly are you? Some kind of bot?"),
    Case(name="mixed_complaint_task", inputs="This report is wrong. Regenerate it with last month's numbers."),
]

dataset: Dataset[str, ClassifyResult] = Dataset(
    name="intent-classifier",
    cases=CASES,
    evaluators=[PrimaryIntentEntailed(), MeanIntentEntailment()],
)


@dataclass
class ScopeMatches(Evaluator[str, ClassifyResult]):
    def evaluate(self, ctx: EvaluatorContext[str, ClassifyResult]) -> EvaluationReason:
        expected = ctx.expected_output
        actual = ctx.output.scope
        return EvaluationReason(
            value=expected is not None and actual is expected.scope,
            reason=(
                f"scope={actual.value!r} "
                f"(expected {expected.scope.value!r})" if expected else "no expectation"
            ),
        )


@dataclass
class RetrievalRouted(Evaluator[str, ClassifyResult]):
    def evaluate(self, ctx: EvaluatorContext[str, ClassifyResult]) -> EvaluationReason:
        expected = ctx.expected_output
        wanted = expected is not None and expected.needs_rag
        return EvaluationReason(
            value=ctx.output.needs_rag == wanted,
            reason=(
                f"needs_rag={ctx.output.needs_rag} rag_query="
                f"{ctx.output.rag_query!r} (expected needs_rag={wanted})"
            ),
        )


def scope_case(name: str, message: str, scope: Scope, needs_rag: bool) -> Case[
    str, ClassifyResult, dict[str, object]
]:
    return Case(
        name=name,
        inputs=message,
        expected_output=ClassifyResult(
            scope=scope, intents=[Intent.QUESTION], needs_rag=needs_rag
        ),
    )


SCOPE_CASES = [
    scope_case("general_knowledge", "What is gravity?", Scope.OUT_OF_SCOPE, False),
    scope_case("arithmetic", "What's 17 * 43?", Scope.OUT_OF_SCOPE, False),
    scope_case("creative", "Write me a haiku about autumn.", Scope.OUT_OF_SCOPE, False),
    scope_case("about_the_demo", "What can you help me with?", Scope.ASSISTANT, False),
    scope_case("greeting", "Hey there", Scope.ASSISTANT, False),
    scope_case(
        "inventory", "What documents do you have for me?", Scope.ASSISTANT, False
    ),
    scope_case(
        "add_a_link",
        "Add https://example.com/docs/guide to my knowledge base",
        Scope.ASSISTANT,
        False,
    ),
    scope_case("clear_material", "Delete everything I've uploaded", Scope.ASSISTANT, False),
    scope_case(
        "named_document",
        "What does the handbook say about releases?",
        Scope.KNOWLEDGE_BASE,
        True,
    ),
    scope_case("person", "Who is Sarah?", Scope.KNOWLEDGE_BASE, True),
    scope_case(
        "vague_this_website",
        "Tell me about this website",
        Scope.KNOWLEDGE_BASE,
        True,
    ),
    scope_case(
        "vague_this_file", "What is this file about?", Scope.KNOWLEDGE_BASE, True
    ),
    scope_case(
        "mixed", "Summarise the handbook, then explain gravity.", Scope.KNOWLEDGE_BASE, True
    ),
]

scope_dataset: Dataset[str, ClassifyResult] = Dataset(
    name="scope-classifier",
    cases=SCOPE_CASES,
    evaluators=[ScopeMatches(), RetrievalRouted()],
)


async def classify_task(message: str) -> ClassifyResult:
    return await classify(message)


def main() -> None:
    load_dotenv()
    report = dataset.evaluate_sync(classify_task)
    report.print(include_input=True, include_output=False, include_reasons=True)
    scope_report = scope_dataset.evaluate_sync(classify_task)
    scope_report.print(include_input=True, include_output=False, include_reasons=True)


if __name__ == "__main__":
    main()
