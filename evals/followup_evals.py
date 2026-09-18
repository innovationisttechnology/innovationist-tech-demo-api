"""Grading the grader.

Which follow-up is best is a matter of taste and not worth asserting. Whether
one should be shown at all is not: the feature's value is that a suggestion
appearing means something, which only holds if weak candidates are actually
refused. These cases fix the candidates and assert the decision.
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from app.ziza_chat.agents.followups import choose_followup

QUESTION = "what does the handbook say about releases?"

ANSWER = (
    "Releases ship every second Tuesday. A release captain named a week ahead "
    "owns the go/no-go call, rotating alphabetically through the platform team."
)


@dataclass
class Candidates:
    questions: list[str]


@dataclass
class ShouldShow(Evaluator[Candidates, str | None]):
    def evaluate(
        self, ctx: EvaluatorContext[Candidates, str | None]
    ) -> EvaluationReason:
        wanted = bool(ctx.expected_output)
        shown = ctx.output is not None
        return EvaluationReason(
            value=shown == wanted,
            reason=(
                f"{'showed ' + repr(ctx.output) if shown else 'refused'} "
                f"(expected {'a suggestion' if wanted else 'nothing'})"
            ),
        )


@dataclass
class ChoseFromTheCandidates(Evaluator[Candidates, str | None]):
    """A grader that rewrites has stopped grading.

    Its text would have been through neither the writer's instructions nor the
    retrieval check that proves the demo can answer it.
    """

    def evaluate(
        self, ctx: EvaluatorContext[Candidates, str | None]
    ) -> EvaluationReason:
        if ctx.output is None:
            return EvaluationReason(value=True, reason="nothing chosen")
        verbatim = ctx.output in ctx.inputs.questions
        return EvaluationReason(
            value=verbatim,
            reason=f"{'verbatim' if verbatim else 'REWRITTEN'}: {ctx.output!r}",
        )


def case(
    name: str, questions: list[str], expected: str | None
) -> Case[Candidates, str | None, dict[str, object]]:
    return Case(name=name, inputs=Candidates(questions), expected_output=expected)


CASES = [
    case(
        "vague_fillers",
        ["Tell me more", "What else is in the handbook?", "Can you elaborate?"],
        None,
    ),
    case(
        "already_answered",
        [
            "When do releases ship?",
            "Who owns the go/no-go call?",
            "How is the release captain chosen?",
        ],
        None,
    ),
    case(
        "restates_the_question",
        ["What does the handbook say about releases?"],
        None,
    ),
    case("nothing_to_choose_from", [], None),
    case(
        "one_genuinely_good",
        ["What happens if the release captain is on leave?"],
        "What happens if the release captain is on leave?",
    ),
    case(
        "good_among_bad",
        [
            "Tell me more",
            "When do releases ship?",
            "What happens if a release is cancelled at go/no-go?",
        ],
        "What happens if a release is cancelled at go/no-go?",
    ),
]

dataset: Dataset[Candidates, str | None] = Dataset(
    name="followup-grader",
    cases=CASES,
    evaluators=[ShouldShow(), ChoseFromTheCandidates()],
)


async def grade_task(candidates: Candidates) -> str | None:
    return await choose_followup(QUESTION, ANSWER, candidates.questions)


def main() -> None:
    load_dotenv()
    report = dataset.evaluate_sync(grade_task)
    report.print(include_input=True, include_output=True, include_reasons=True)


if __name__ == "__main__":
    main()
