"""Grading primitives, reimplemented on Inspect's public API only.

`model_graded_qa`/`model_graded_fact` are public, but their grading internals
(`DEFAULT_GRADE_PATTERN`, the default template, `model_scoring_prompt`) live in
the internal module `inspect_ai.scorer._model` and only run inside the live
`score(state, target)` path during `eval()`. Re-grading a *stored* completion
needs none of that machinery and cannot reach it via the public API, so we
vendor a small grading template + parser here — the same discipline
`inspect-claim-support` used when it reimplemented `chat_history`.
"""

import re

# QA-style rubric mirroring Inspect's own model-graded family, so the re-grade
# uses the same C/P/I verdict surface the original eval's grader did. The judge
# model is the only thing that varies between grader A and grader B; the prompt
# construction is held identical for both.
DEFAULT_DRIFT_TEMPLATE = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

Does the submission meet the criterion?

After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes)
where LETTER is one of C, P, I. Choose ONE option: "C" for correct answers, "P" for
partially correct answers, or "I" for incorrect answers.

First, write out in a step by step manner your reasoning about the criterion. Then,
end with your answer formatted as 'GRADE: $LETTER' where LETTER is one of C, P, I.
""".strip()

# Copied VERBATIM from inspect_ai 0.3.242 `scorer/_model.py` (DEFAULT_GRADE_PATTERN)
# for parse parity with upstream grading. Reproducing a regex constant (not an
# internal symbol) keeps the no-internal-imports discipline intact. Properties:
# greedy leading `.*` (DOTALL) binds to the *final* GRADE: so earlier or injected
# mentions cannot win; `(?<!\w)GRADE(?!\w)` rejects GRID/GRAND/XGRADE; the
# zero-width-unicode escapes (\u200b … \ufeff) tolerate graders that emit invisible
# codepoints around the colon; captures one of [CPI]. Stored as escape text (raw
# string) — byte-identical to upstream's distributed form and ASCII-reviewable.
DEFAULT_GRADE_PATTERN = r"(?is).*(?<!\w)GRADE(?!\w)[\s\u200b\u200c\u200d\u200e\u200f\u2060\u2063\ufeff]*:[\s\u200b\u200c\u200d\u200e\u200f\u2060\u2063\ufeff]*([CPI])"

# Known template slots. Substituted in a single pass (see build_prompt).
_SLOT_RE = re.compile(r"\{(question|answer|criterion)\}")


def build_prompt(template: str, *, question: str, answer: str, criterion: str) -> str:
    """Fill the grading template's slots in a single pass.

    Deliberately not `str.format` (a literal `{` in LLM-produced `answer` would
    raise / allow brace injection) and deliberately not chained `str.replace`
    either: chaining `.replace("{answer}", answer).replace("{criterion}", ...)`
    lets an `answer` that itself contains the literal text `{criterion}` get
    cross-substituted by the later step. `re.sub` with a function replacement
    scans the template once and never re-scans inserted text, so dataset/MUT
    content can never inject into another slot.
    """
    repl = {"question": question, "answer": answer, "criterion": criterion}
    return _SLOT_RE.sub(lambda m: repl[m.group(1)], template)


def parse_grade(completion: str, pattern: str = DEFAULT_GRADE_PATTERN) -> str | None:
    """Extract the verdict letter (C/P/I) from a grader completion.

    Returns the upper-cased capture group, or ``None`` when no verdict can be
    parsed. ``None`` is the instrument-failure signal that the caller turns into
    ``Score.unscored()`` — never a fabricated INCORRECT.
    """
    match = re.search(pattern, completion)
    if match:
        return match.group(1).upper()
    return None
