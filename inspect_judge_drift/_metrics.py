"""Dependency-free drift metrics over (grader_a, grader_b) verdict pairs.

Each pair is ``(value_a, value_b)`` where a value is a verdict label (e.g. "C")
or ``None`` when that grader failed to produce a parseable verdict. A pair is
*comparable* only when both sides are non-``None``. Unscored pairs are excluded
from the drift-rate and kappa denominators and reported separately by
`unscored_rate` — the same "don't hide instrument failure in the denominator"
principle argued upstream in inspect_ai#4026 / #4048.
"""

from collections import Counter
from typing import Optional, Sequence, Tuple

Pair = Tuple[Optional[str], Optional[str]]


def _comparable(pairs: Sequence[Pair]) -> list[Pair]:
    return [(a, b) for a, b in pairs if a is not None and b is not None]


def drift_rate(pairs: Sequence[Pair]) -> Optional[float]:
    """Fraction of *comparable* samples where the two graders disagree.

    Returns ``None`` when there are no comparable samples (empty log, or every
    sample lost to a parse failure) — an honest "undefined", not a misleading
    0.0.
    """
    comparable = _comparable(pairs)
    if not comparable:
        return None
    disagree = sum(1 for a, b in comparable if a != b)
    return disagree / len(comparable)


def cohens_kappa(pairs: Sequence[Pair]) -> Optional[float]:
    """Cohen's kappa between grader A and grader B over comparable samples.

    κ = (p_o - p_e) / (1 - p_e), with p_o the observed agreement and p_e the
    agreement expected from each grader's marginal verdict distribution.
    Returns ``None`` when there are no comparable samples. Also returns ``None``
    when p_e == 1 — i.e. both graders placed every comparable sample in the same
    single category: κ is then 0/0, undefined — there is no variance to assess
    agreement beyond chance, so reporting a number would overstate the result.
    The companion ``drift_rate`` (0.0 in that case) carries "no observed
    disagreement" honestly, without κ pretending to measure agreement.
    """
    comparable = _comparable(pairs)
    n = len(comparable)
    if n == 0:
        return None
    p_o = sum(1 for a, b in comparable if a == b) / n
    count_a = Counter(a for a, _ in comparable)
    count_b = Counter(b for _, b in comparable)
    categories = set(count_a) | set(count_b)
    p_e = sum((count_a[k] / n) * (count_b[k] / n) for k in categories)
    if p_e >= 1.0:
        return None
    return (p_o - p_e) / (1.0 - p_e)


def unscored_rate(pairs: Sequence[Pair]) -> Optional[float]:
    """Fraction of all samples where at least one grader failed to parse.

    Returns ``None`` for an empty input.
    """
    if not pairs:
        return None
    unscored = sum(1 for a, b in pairs if a is None or b is None)
    return unscored / len(pairs)
