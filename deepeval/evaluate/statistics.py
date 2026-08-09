"""Sampling-uncertainty helpers for aggregate evaluation results.

A pass rate is an estimate from a finite number of test cases, not an exact
property of the system under test. With 5 test cases, "80% pass rate" and "100%
pass rate" are barely distinguishable; with 500 they are worlds apart. These
helpers put the sample size back into the reported number.
"""

import math

# Two-sided 95% standard-normal quantile. Hardcoded so this module stays
# dependency-free (no scipy) and deterministic.
Z_95 = 1.959963984540054


def wilson_score_interval(
    successes: int, total: int, z: float = Z_95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as fractions in [0, 1].

    Preferred over the textbook normal (Wald) interval because Wald collapses to
    zero width at 0% and 100% — exactly the cases an eval suite hits most often —
    and undercovers badly for small samples.

    Args:
        successes: number of passing verdicts.
        total: number of verdicts (passes + fails).
        z: standard-normal quantile; defaults to a two-sided 95% interval.

    Returns:
        (low, high) as fractions of 1.

    Raises:
        ValueError: if `total` is not positive or `successes` is out of range.
    """
    if total <= 0:
        raise ValueError("total must be positive to form an interval")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")

    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        p * (1 - p) / total + z2 / (4 * total * total)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def format_pass_rate_with_interval(
    successes: int, total: int, z: float = Z_95
) -> str:
    """'80.00% (95% CI 37.6-96.4%)' — a pass rate that shows what the sample supports."""
    rate = (successes / total) * 100
    low, high = wilson_score_interval(successes, total, z)
    return f"{rate:.2f}% (95% CI {low * 100:.1f}-{high * 100:.1f}%)"
