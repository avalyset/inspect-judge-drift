"""inspect-judge-drift — measure LLM-judge verdict drift across model versions.

Re-grade a stored Inspect `.eval` log with two grader configurations over the
*same* logged samples, isolating the judge model as the only variable, and
report the drift rate and Cohen's kappa between them.
"""

from ._drift import (
    DriftReport,
    GraderSpec,
    SampleDrift,
    regrade_eval_log,
    regrade_eval_log_async,
)
from ._grading import (
    DEFAULT_DRIFT_TEMPLATE,
    DEFAULT_GRADE_PATTERN,
    build_prompt,
    parse_grade,
)
from ._metrics import cohens_kappa, drift_rate, unscored_rate

__all__ = [
    "regrade_eval_log",
    "regrade_eval_log_async",
    "GraderSpec",
    "DriftReport",
    "SampleDrift",
    "drift_rate",
    "cohens_kappa",
    "unscored_rate",
    "parse_grade",
    "build_prompt",
    "DEFAULT_DRIFT_TEMPLATE",
    "DEFAULT_GRADE_PATTERN",
]

__version__ = "0.1.0"
