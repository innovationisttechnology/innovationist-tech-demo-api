"""Grading which documents get opening questions, and which get none.

A document made of questions — a call script, a form, an interview guide —
scores perfectly against retrieval and answers nothing, so the check that
matters is whether the grader refuses it. Fixed documents, fixed candidates,
graded decision.
"""

from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from app.ziza_chat.agents.starter_questions import keep_answered

CALL_SCRIPT = """\
Kindergarten Waiver Call Script

Calling: Wayland Union Schools district office, (269) 792-2181
Best time to call: weekday mornings, 8:00-10:00am

Questions to ask
1. Is there a deadline for the 2026-2027 school year I need to meet?
2. What documentation is required to submit a waiver request?
3. Who makes the final decision, and how long does it take?

Before you hang up
- Note the exact waiver deadline given.
- Note any documents they say are required.
- Get the name of the person you spoke with.
"""

POLICY = """\
Wayland Union Schools - Kindergarten Waiver Policy

A child who turns five between September 2 and December 1 may enrol with an age
waiver. Requests for the 2026-2027 school year must be submitted by 1 August
2026. A completed readiness screening and a copy of the birth certificate are
required with every request. The superintendent decides within ten working
days.
"""


@dataclass
class Grading:
    document: str
    text: str
    candidates: list[str]


@dataclass
class KeptTheRightOnes(Evaluator[Grading, list[str]]):
    def evaluate(self, ctx: EvaluatorContext[Grading, list[str]]) -> EvaluationReason:
        expected = set(ctx.expected_output or [])
        kept = set(ctx.output or [])
        return EvaluationReason(
            value=kept == expected,
            reason=f"kept {sorted(kept)} (expected {sorted(expected)})",
        )


@dataclass
class InventedNothing(Evaluator[Grading, list[str]]):
    def evaluate(self, ctx: EvaluatorContext[Grading, list[str]]) -> EvaluationReason:
        offered = set(ctx.inputs.candidates)
        stray = sorted(set(ctx.output or []) - offered)
        return EvaluationReason(
            value=not stray,
            reason="verbatim" if not stray else f"REWRITTEN: {stray}",
        )


ASKED_IN_THE_SCRIPT = [
    "What is the waiver deadline for the 2026-2027 school year?",
    "What documentation is required to submit a waiver request?",
]

ANSWERED_BY_THE_POLICY = [
    "What is the deadline for submitting a waiver request for 2026-2027?",
    "What documents must be included with a waiver request?",
]

CASES: list[Case[Grading, list[str], Any]] = [
    Case(
        name="script_answers_none_of_its_own_questions",
        inputs=Grading("Call_Script.docx", CALL_SCRIPT, ASKED_IN_THE_SCRIPT),
        expected_output=[],
    ),
    Case(
        name="script_does_answer_what_it_states",
        inputs=Grading(
            "Call_Script.docx",
            CALL_SCRIPT,
            ["What number do I call?", "When is the best time to call?"],
        ),
        expected_output=["What number do I call?", "When is the best time to call?"],
    ),
    Case(
        name="policy_answers_the_same_questions",
        inputs=Grading("Waiver_Policy.pdf", POLICY, ANSWERED_BY_THE_POLICY),
        expected_output=ANSWERED_BY_THE_POLICY,
    ),
    Case(
        name="policy_refuses_what_it_does_not_cover",
        inputs=Grading(
            "Waiver_Policy.pdf",
            POLICY,
            ["Is there an appeal process if the waiver is denied?"],
        ),
        expected_output=[],
    ),
    Case(
        name="nothing_to_grade",
        inputs=Grading("Waiver_Policy.pdf", POLICY, []),
        expected_output=[],
    ),
]

dataset: Dataset[Grading, list[str]] = Dataset(
    name="starter-question-grader",
    cases=CASES,
    evaluators=[KeptTheRightOnes(), InventedNothing()],
)


async def grade_task(grading: Grading) -> list[str]:
    return await keep_answered(grading.document, grading.text, grading.candidates)


def main() -> None:
    load_dotenv()
    report = dataset.evaluate_sync(grade_task)
    report.print(include_input=False, include_output=True, include_reasons=True)


if __name__ == "__main__":
    main()
