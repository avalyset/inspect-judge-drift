import math

import pytest

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentText, ModelOutput, get_model

from inspect_judge_drift import (
    GraderSpec,
    build_prompt,
    cohens_kappa,
    drift_rate,
    parse_grade,
    regrade_eval_log,
    unscored_rate,
)


def _mock(*texts: str):
    """A mockllm model that returns the given completions in order."""
    return get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm/model", [ContentText(text=t)])
            for t in texts
        ],
    )


def _log(n: int):
    """Produce an in-memory eval log with ``n`` samples (subject = mock)."""
    dataset = [
        Sample(input=f"Task {i}: meets criterion?", target="C") for i in range(n)
    ]
    subject = _mock(*[f"subject answer {i}" for i in range(n)])
    task = Task(dataset=dataset, scorer=[])
    return eval(task, model=subject)[0]


# --------------------------------------------------------------------------
# Pure metric tests (no model) — deterministic, order-independent.
# --------------------------------------------------------------------------


def test_drift_rate_basic():
    # 4 comparable pairs, 1 disagreement.
    pairs = [("C", "C"), ("C", "I"), ("P", "P"), ("I", "I")]
    assert drift_rate(pairs) == pytest.approx(0.25)


def test_drift_rate_excludes_unscored_from_denominator():
    # 2 comparable (1 disagreement), 2 with a None side. Denominator must be 2,
    # not 4 — the parse failures are not silently counted as agreement.
    pairs = [("C", "I"), ("C", "C"), (None, "C"), ("P", None)]
    assert drift_rate(pairs) == pytest.approx(0.5)
    assert unscored_rate(pairs) == pytest.approx(0.5)


def test_drift_rate_empty_is_none():
    assert drift_rate([]) is None
    assert drift_rate([(None, "C"), ("C", None)]) is None  # no comparable pairs
    assert unscored_rate([]) is None


def test_cohens_kappa_perfect_agreement():
    pairs = [("C", "C"), ("I", "I"), ("P", "P"), ("C", "C")]
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_cohens_kappa_single_category_is_none():
    # Both graders constant on the same category → p_e == 1 → κ is 0/0, undefined.
    # Returns None (not 1.0): there is no variance to assess agreement beyond
    # chance. drift_rate carries "no observed disagreement" separately.
    assert cohens_kappa([("C", "C"), ("C", "C")]) is None
    assert drift_rate([("C", "C"), ("C", "C")]) == pytest.approx(0.0)


def test_cohens_kappa_known_value():
    # Hand-computed 2x2: agreements on diagonal.
    # A: C,C,C,I,I,I  B: C,C,I,I,I,C  -> p_o = 4/6.
    # marginals A: C=3,I=3 ; B: C=3,I=3 -> p_e = .25+.25 = .5
    # kappa = (4/6 - .5)/(1-.5) = (0.6667-0.5)/0.5 = 0.3333
    pairs = [("C", "C"), ("C", "C"), ("C", "I"), ("I", "I"), ("I", "I"), ("I", "C")]
    assert cohens_kappa(pairs) == pytest.approx(1 / 3, abs=1e-9)


def test_cohens_kappa_chance_agreement_near_zero():
    # A constant C, B half C half I → observed agreement equals chance → kappa 0.
    pairs = [("C", "C"), ("C", "I"), ("C", "C"), ("C", "I")]
    assert cohens_kappa(pairs) == pytest.approx(0.0, abs=1e-9)


def test_cohens_kappa_empty_is_none():
    assert cohens_kappa([]) is None
    assert cohens_kappa([(None, "C")]) is None


# --------------------------------------------------------------------------
# Grading primitive tests.
# --------------------------------------------------------------------------


def test_parse_grade_binds_to_last():
    # Earlier GRADE mentions (chain-of-thought / injected) must not win.
    assert parse_grade("I first thought GRADE: I but actually\nGRADE: C") == "C"
    assert parse_grade("no verdict at all") is None


def test_build_prompt_single_pass_no_cross_injection():
    # An answer containing the literal text "{criterion}" must NOT be
    # substituted by the criterion slot (the failure mode of chained .replace),
    # and a literal "{" must not raise (the failure mode of str.format).
    out = build_prompt(
        "Q={question} A={answer} C={criterion}",
        question="q",
        answer="see {criterion} and a stray { brace",
        criterion="SECRET",
    )
    assert "A=see {criterion} and a stray { brace" in out
    assert "C=SECRET" in out
    assert out.count("SECRET") == 1  # criterion injected exactly once


# --------------------------------------------------------------------------
# Integration tests: re-grade a real in-memory log with two mock graders.
# Assertions are aggregate and order-independent (eval may run concurrently).
# --------------------------------------------------------------------------


def test_regrade_determinism_identical_graders():
    log = _log(3)
    # Both graders emit the SAME varied verdict sequence. Re-grade processes
    # samples in a stable order, calling grader A then grader B on each sample,
    # so identical output lists agree per-sample → zero drift and (because the
    # verdicts span >1 category) κ == 1.0. Using varied verdicts here genuinely
    # exercises the κ == 1.0 path; constant verdicts would make κ undefined.
    report = regrade_eval_log(
        log,
        GraderSpec(_mock("GRADE: C", "GRADE: I", "GRADE: P"), label="A"),
        GraderSpec(_mock("GRADE: C", "GRADE: I", "GRADE: P"), label="B"),
    )
    assert report.n_total == 3
    assert report.n_comparable == 3
    assert report.drift_rate == pytest.approx(0.0)
    assert report.cohens_kappa == pytest.approx(1.0)
    assert report.unscored_rate == pytest.approx(0.0)


def test_regrade_detects_drift():
    log = _log(3)
    # Grader A constant C; grader B emits exactly one I among its three outputs.
    # Since A is always C, drift occurs iff B says I → exactly one disagreement
    # regardless of sample/processing order → drift_rate == 1/3.
    report = regrade_eval_log(
        log,
        _mock("GRADE: C", "GRADE: C", "GRADE: C"),
        _mock("GRADE: C", "GRADE: I", "GRADE: C"),
    )
    assert report.n_comparable == 3
    assert report.drift_rate == pytest.approx(1 / 3)
    # A constant → no marginal signal → kappa 0.
    assert report.cohens_kappa == pytest.approx(0.0, abs=1e-9)


def test_regrade_parse_fail_is_unscored_not_drift():
    log = _log(3)
    # Grader A fails to emit a verdict on exactly one sample; B always says C.
    report = regrade_eval_log(
        log,
        _mock("GRADE: C", "no parseable verdict here", "GRADE: C"),
        _mock("GRADE: C", "GRADE: C", "GRADE: C"),
    )
    assert report.n_total == 3
    assert report.n_unscored == 1
    assert report.n_comparable == 2
    assert report.unscored_rate == pytest.approx(1 / 3)
    # The two comparable samples both agree (C vs C) → no fabricated drift.
    assert report.drift_rate == pytest.approx(0.0)
    # The unscored sample carries a NaN Score.unscored() on the failing grader.
    unscored = [s for s in report.samples if s.unscored]
    assert len(unscored) == 1
    assert math.isnan(unscored[0].score_a.value)
    assert unscored[0].score_a.metadata["grader_error"] == "parse_fail"


def test_regrade_empty_log():
    log = _log(1)
    log.samples = []  # simulate a log with no samples
    report = regrade_eval_log(log, _mock(), _mock())
    assert report.n_total == 0
    assert report.drift_rate is None
    assert report.cohens_kappa is None
    assert report.unscored_rate is None
