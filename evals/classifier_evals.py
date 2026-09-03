from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from app.ziza_chat.agents.outputs import ClassifyResult, Intent
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


async def classify_task(message: str) -> ClassifyResult:
    return await classify(message)


def main() -> None:
    load_dotenv()
    report = dataset.evaluate_sync(classify_task)
    report.print(include_input=True, include_output=False, include_reasons=True)


if __name__ == "__main__":
    main()
