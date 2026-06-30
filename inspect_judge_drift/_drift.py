"""Re-grade a stored Inspect eval log with two graders and measure judge drift.

The premise this package stands on (verified against `inspect_ai.log`): an
`EvalSample` carries `output.completion` (the MUT output the grader judges),
`target`, and `input`. So re-grading reads those three fields and runs two
grader configurations over *exactly the same stored inputs* — the judge model
is the only variable. No MUT re-run, byte-stable input.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.model import Model, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, PARTIAL, Score

from ._grading import (
    DEFAULT_DRIFT_TEMPLATE,
    DEFAULT_GRADE_PATTERN,
    build_prompt,
    parse_grade,
)
from ._metrics import Pair, cohens_kappa, drift_rate, unscored_rate

# Verdict letter → Inspect Score value. The letters already equal the public
# constants (CORRECT == "C", PARTIAL == "P", INCORRECT == "I"); the mapping is
# explicit so a custom rubric/pattern can be adapted in one place.
_VALUE = {"C": CORRECT, "P": PARTIAL, "I": INCORRECT}


@dataclass
class GraderSpec:
    """A grader configuration: a model plus an optional display label.

    ``model`` may be a model name (resolved lazily, once, at re-grade time — not
    at construction) or an already-constructed `Model` (e.g. a mock for tests).
    """

    model: Union[str, Model]
    label: Optional[str] = None

    @property
    def display(self) -> str:
        if self.label:
            return self.label
        return self.model if isinstance(self.model, str) else repr(self.model)


@dataclass
class SampleDrift:
    """Per-sample re-grade outcome for the two graders."""

    sample_id: Any
    value_a: Optional[str]
    value_b: Optional[str]
    agree: Optional[bool]  # None when the sample is not comparable (a parse fail)
    unscored: bool
    score_a: Score
    score_b: Score

    @property
    def pair(self) -> Pair:
        return (self.value_a, self.value_b)


@dataclass
class DriftReport:
    """Aggregate drift between two graders over a re-graded log."""

    samples: list[SampleDrift]
    grader_a: str
    grader_b: str

    @property
    def pairs(self) -> list[Pair]:
        return [s.pair for s in self.samples]

    @property
    def n_total(self) -> int:
        return len(self.samples)

    @property
    def n_comparable(self) -> int:
        return sum(1 for s in self.samples if not s.unscored)

    @property
    def n_unscored(self) -> int:
        return sum(1 for s in self.samples if s.unscored)

    @property
    def drift_rate(self) -> Optional[float]:
        return drift_rate(self.pairs)

    @property
    def cohens_kappa(self) -> Optional[float]:
        return cohens_kappa(self.pairs)

    @property
    def unscored_rate(self) -> Optional[float]:
        return unscored_rate(self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grader_a": self.grader_a,
            "grader_b": self.grader_b,
            "n_total": self.n_total,
            "n_comparable": self.n_comparable,
            "n_unscored": self.n_unscored,
            "drift_rate": self.drift_rate,
            "cohens_kappa": self.cohens_kappa,
            "unscored_rate": self.unscored_rate,
        }


def _input_text(sample: EvalSample) -> str:
    inp = sample.input
    if isinstance(inp, str):
        return inp
    parts: list[str] = []
    for message in inp:
        text = getattr(message, "text", None)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _target_text(sample: EvalSample) -> str:
    target = sample.target
    if isinstance(target, str):
        return target
    if isinstance(target, list):
        return "\n".join(target)
    return str(target)


def _to_score(value: Optional[str], completion: str, prompt: str, answer: str) -> Score:
    if value is None:
        # Grade-parse failure: the grader (the scoring instrument) failed, which
        # is not a "no answer" from the model under test. Leave the sample
        # unscored — out of the accuracy denominator — rather than fabricating an
        # INCORRECT. This embodies the position argued in inspect_ai#4026/#4048.
        return Score.unscored(
            answer=answer,
            explanation=completion,
            metadata={"grader_error": "parse_fail", "grader_prompt": prompt},
        )
    return Score(
        value=_VALUE[value],
        answer=answer,
        explanation=completion,
        metadata={"grading": value, "grader_prompt": prompt},
    )


async def regrade_eval_log_async(
    log: Union[str, Path, EvalLog],
    grader_a: Union[GraderSpec, str, Model],
    grader_b: Union[GraderSpec, str, Model],
    *,
    template: Optional[str] = None,
    grade_pattern: Optional[str] = None,
) -> DriftReport:
    """Re-grade every sample in ``log`` with two graders and report drift.

    The same prompt — built from the stored ``input`` / ``output.completion`` /
    ``target`` — is sent to both graders, so the grader model is the only
    variable. A grade-parse failure on either side makes the sample unscored
    (excluded from drift_rate and kappa, counted in unscored_rate).
    """
    if isinstance(log, (str, Path)):
        log = read_eval_log(str(log))
    samples = log.samples or []

    spec_a = grader_a if isinstance(grader_a, GraderSpec) else GraderSpec(grader_a)
    spec_b = grader_b if isinstance(grader_b, GraderSpec) else GraderSpec(grader_b)

    # Lazy resolution: get_model() is called here (re-grade time), not when the
    # GraderSpec was constructed. Resolve once and reuse across samples.
    model_a = spec_a.model if isinstance(spec_a.model, Model) else get_model(spec_a.model)
    model_b = spec_b.model if isinstance(spec_b.model, Model) else get_model(spec_b.model)

    tmpl = template or DEFAULT_DRIFT_TEMPLATE
    pattern = grade_pattern or DEFAULT_GRADE_PATTERN

    results: list[SampleDrift] = []
    for sample in samples:
        answer = sample.output.completion if sample.output else ""
        prompt = build_prompt(
            tmpl,
            question=_input_text(sample),
            answer=answer,
            criterion=_target_text(sample),
        )

        out_a = await model_a.generate(prompt)
        out_b = await model_b.generate(prompt)
        value_a = parse_grade(out_a.completion, pattern)
        value_b = parse_grade(out_b.completion, pattern)

        unscored = value_a is None or value_b is None
        agree = None if unscored else (value_a == value_b)
        results.append(
            SampleDrift(
                sample_id=sample.id,
                value_a=value_a,
                value_b=value_b,
                agree=agree,
                unscored=unscored,
                score_a=_to_score(value_a, out_a.completion, prompt, answer),
                score_b=_to_score(value_b, out_b.completion, prompt, answer),
            )
        )

    return DriftReport(samples=results, grader_a=spec_a.display, grader_b=spec_b.display)


def regrade_eval_log(
    log: Union[str, Path, EvalLog],
    grader_a: Union[GraderSpec, str, Model],
    grader_b: Union[GraderSpec, str, Model],
    *,
    template: Optional[str] = None,
    grade_pattern: Optional[str] = None,
) -> DriftReport:
    """Synchronous wrapper around `regrade_eval_log_async`.

    Convenience for scripts and notebooks. Inside an already-running event loop,
    call `regrade_eval_log_async` directly instead.
    """
    return asyncio.run(
        regrade_eval_log_async(
            log,
            grader_a,
            grader_b,
            template=template,
            grade_pattern=grade_pattern,
        )
    )
