"""Statistical helpers for aggregate evaluation reporting."""

import math
from typing import Optional, Tuple

# Two-sided standard normal critical values.
_Z_SCORES = {
    0.80: 1.2816,
    0.90: 1.6449,
    0.95: 1.9600,
    0.99: 2.5758,
}


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation (Wald) interval, which can
    produce bounds outside [0, 1] and collapses to a single point when
    all observations succeed or all fail. Both cases are common in
    evaluation suites, which are typically small and tuned so that most
    test cases pass.

    Note this describes sampling uncertainty only: it reflects that the
    suite is a sample of possible inputs, not the run-to-run variance of
    a nondeterministic judge model.

    Args:
        successes: Number of passing observations.
        n: Total number of observations with a verdict.
        confidence: Confidence level. One of 0.80, 0.90, 0.95, 0.99.

    Returns:
        (lower, upper) as proportions in [0, 1], or None if n <= 0.

    Raises:
        ValueError: If confidence is unsupported or successes is invalid.
    """
    if n <= 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError(
            f"'successes' must be between 0 and {n}, got {successes}"
        )

    z = _Z_SCORES.get(confidence)
    if z is None:
        raise ValueError(
            f"Unsupported confidence level: {confidence}. "
            f"Expected one of {sorted(_Z_SCORES)}."
        )

    p = successes / n
    z_sq = z * z

    denominator = 1 + z_sq / n
    center = (p + z_sq / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z_sq / (4 * n * n))) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))
