# inspect-judge-drift

[![ci](https://github.com/avalyset/inspect-judge-drift/actions/workflows/ci.yml/badge.svg)](https://github.com/avalyset/inspect-judge-drift/actions/workflows/ci.yml)

Measure **LLM-judge verdict drift across model versions** for
[Inspect AI](https://inspect.aisi.org.uk/).

When the model you use as a grader is upgraded (say `claude-opus-4-7` →
`claude-opus-4-8`), do its verdicts change on identical inputs? `inspect-judge-drift`
answers that by **re-grading a stored `.eval` log** with two grader
configurations over the *same logged samples* and reporting how often they
disagree.

## Why re-grade from a stored log (and not run two live evals)

Drift measurement requires holding everything except the judge constant. Running
two live evals introduces two fresh variance sources — the model-under-test's
own stochasticity between runs **and** grader sampling — that contaminate the
exact signal you are trying to isolate. Re-grading a single stored log fixes the
inputs byte-for-byte: every grader sees the same `output.completion`, `target`,
and `input` that were already recorded. **The judge model is the only variable.**
No MUT re-run, reproducible input.

This is the input a credible power-dimensioned drift study has to stand on. Live
dual-grading measures something weaker and costs more.

## How it isolates the judge

For each sample, the same grading prompt — built from the stored
`input` / `output.completion` / `target` — is sent to **both** graders. The only
difference between the two verdicts is the grader model.

A grade-parse failure (the grader not emitting a parseable verdict) is treated
as a **scoring-instrument failure**: that sample is left `Score.unscored()`,
excluded from the drift-rate and kappa denominators, and reported separately as
`unscored_rate` — never fabricated into a verdict. This mirrors the position
argued upstream in
[inspect_ai#4026](https://github.com/UKGovernmentBEIS/inspect_ai/issues/4026) /
[#4048](https://github.com/UKGovernmentBEIS/inspect_ai/pull/4048): an instrument
failure must stay visible, not masquerade as a model result.

## Install

```bash
pip install inspect-judge-drift
```

## Use

```python
from inspect_judge_drift import regrade_eval_log, GraderSpec

report = regrade_eval_log(
    "logs/2026-06-25T03-57-26-00-00_task_RXYpgDBbXfECH6EoqaX5w9.eval",
    GraderSpec("anthropic/claude-opus-4-7", label="opus-4.7"),
    GraderSpec("anthropic/claude-opus-4-8", label="opus-4.8"),
)

print(report.to_dict())
# Illustrative output (hypothetical numbers, not a measured result):
# {
#   'grader_a': 'opus-4.7', 'grader_b': 'opus-4.8',
#   'n_total': 100, 'n_comparable': 98, 'n_unscored': 2,
#   'drift_rate': 0.061,        # 6.1% of comparable samples flipped verdict
#   'cohens_kappa': 0.83,       # grader-vs-grader agreement
#   'unscored_rate': 0.02,      # 2% lost to grade-parse failure (reported, not hidden)
# }

for s in report.samples:
    if s.agree is False:
        print(s.sample_id, s.value_a, "->", s.value_b)
```

Inside an existing event loop (a notebook, an async app), use the async form:

```python
from inspect_judge_drift import regrade_eval_log_async
report = await regrade_eval_log_async(log, grader_a, grader_b)
```

## What it reports

| Field | Meaning |
| --- | --- |
| `drift_rate` | fraction of **comparable** samples where the two graders disagree (`None` if none are comparable) |
| `cohens_kappa` | Cohen's κ between grader A and grader B over comparable samples |
| `unscored_rate` | fraction of samples where at least one grader failed to parse |
| `n_total` / `n_comparable` / `n_unscored` | sample counts |

Parse failures are excluded from `drift_rate` / `cohens_kappa` and surfaced in
`unscored_rate` instead — the denominator is never silently inflated.

## Fidelity boundary (honest limit)

The grade-extraction pattern (`DEFAULT_GRADE_PATTERN`) is reproduced **verbatim**
from Inspect's internal `scorer/_model.py`, including its word-boundary guard and
zero-width-unicode tolerance, so verdict parsing matches upstream character for
character. But the package depends on **no internal Inspect module**, which means
it cannot call upstream's internal `neutralize_structural_delimiters` prompt
preparation (it is private). So the grading *prompt* is a faithful reimplementation,
**not byte-identical** to what `model_graded_qa` builds at eval time.

The consequence is bounded and disclosed:

- For **A-vs-B drift** — the package's actual job — both graders receive the
  *exact same* reimplemented prompt, so any prompt-prep difference from upstream
  is held constant and cancels. The drift signal is internally consistent.
- For the stronger claim "this reproduces Inspect's grading byte-for-byte" — it
  does not, and does not pretend to. Drift is measured under this faithful-but-
  not-identical reimplementation.

This is the same kind of fidelity limit `inspect-claim-support` flagged for its
grader path — stated plainly rather than left implicit.

## Scope (0.1.0)

- **Log mode only**: re-grade from a stored `.eval`. Live dual-grading is a
  possible later option, deliberately not in 0.1.0 — it measures a weaker,
  noisier signal.
- Default rubric is the C/P/I model-graded family; pass `template=` and
  `grade_pattern=` for a custom rubric.
- Built on Inspect's **public API only** (`read_eval_log`, `EvalSample`,
  `Score`, `get_model`). The internal grading template and pattern are
  reimplemented locally rather than imported, so the package depends on no
  internal Inspect module.

## Where it sits next to related tools

The tool *enables* a drift study; it is not the study. A credible study still
needs a power-dimensioned N, a drift-benchmark dataset, and published findings.
Its axis is distinct from the neighbouring eval-reliability work: this measures
**drift in the judge across versions, with the MUT held constant** — not
confidence intervals on a single mean, not deltas between models-under-test, not
prompt-output snapshot regression.

## Citation

See [`CITATION.cff`](CITATION.cff).
