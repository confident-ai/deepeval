"""Sampling-uncertainty helpers for aggregate evaluation results.

A pass rate is an estimate from a finite number of test cases, not an exact
property of the system under test. With 5 test cases, "80% pass rate" and "100%
pass rate" are barely distinguishable; with 500 they are worlds apart. These
helpers put the sample size back into the reported number.
"""

import math
from statistics import NormalDist

_STANDARD_NORMAL = NormalDist()


def z_for_confidence(confidence: float = 0.95) -> float:
    """Two-sided standard-normal quantile for a confidence level.

    `confidence=0.95` returns 1.9599639845..., the familiar 1.96. Derived from
    the stdlib rather than hardcoded so the level a caller asks for and the
    level the arithmetic uses cannot drift apart.

    Raises:
        ValueError: if `confidence` is not strictly between 0 and 1.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return _STANDARD_NORMAL.inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as fractions in [0, 1].

    Preferred over the textbook normal (Wald) interval because Wald collapses to
    zero width at 0% and 100% — exactly the cases an eval suite hits most often —
    and undercovers badly for small samples.

    Args:
        successes: number of passing verdicts.
        total: number of verdicts (passes + fails).
        confidence: two-sided confidence level, e.g. 0.95 or 0.99.

    Returns:
        (low, high) as fractions of 1.

    Raises:
        ValueError: if `total` is not positive, `successes` is out of range, or
            `confidence` is not strictly between 0 and 1.
    """
    if total <= 0:
        raise ValueError("total must be positive to form an interval")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")

    z = z_for_confidence(confidence)
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        p * (1 - p) / total + z2 / (4 * total * total)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def format_pass_rate_with_interval(
    successes: int, total: int, confidence: float = 0.95
) -> str:
    """'80.00% (95% CI 37.6-96.4%)' — a pass rate that shows what the sample supports.

    The label reports the level the interval was actually computed at, so
    `confidence=0.99` prints "99% CI" rather than a stale "95%".
    """
    rate = (successes / total) * 100
    low, high = wilson_score_interval(successes, total, confidence)
    label = f"{confidence * 100:g}%"
    return f"{rate:.2f}% ({label} CI {low * 100:.1f}-{high * 100:.1f}%)"
