"""Reliability statistics for repeated measurements of a single rater.

These are the standard intra-rater (test-retest) quantities, implemented on
the Python standard library so the metric adds no dependency: `deepeval` does
not depend on `numpy`, and every `numpy` use in the package is a deferred
import inside an optional code path.

Replicates are treated throughout as a sample drawn from the judge's own
sampling distribution — that distribution is exactly what is being estimated —
so the spread uses the sample standard deviation, and the interval on the mean
uses the nonparametric bootstrap.

References for the methods used here:

- Nonparametric bootstrap and the percentile interval: Efron, B. and
  Tibshirani, R. (1993), *An Introduction to the Bootstrap*, Chapman & Hall,
  chapters 6 and 13.
- Test-retest reliability and pairwise agreement: Shrout, P. E. and Fleiss,
  J. L. (1979), "Intraclass correlations: uses in assessing rater
  reliability", *Psychological Bulletin* 86(2), 420-428.
"""

from __future__ import annotations

import random
import statistics
from typing import List, Optional, Sequence, Tuple


def stability(
    scores: Sequence[float], score_range: Tuple[float, float] = (0.0, 1.0)
) -> float:
    """Return ``1 - normalized spread`` of ``scores``, clamped to [0, 1].

    Spread is the sample standard deviation divided by half the width of
    ``score_range``. Half the width is the largest standard deviation a
    *population* on that range can have, attained when half the mass sits at
    each end, so the ratio puts the spread on a 0-1 scale.

    ``1.0`` means every repeat produced the same score. ``0.0`` means the
    repeats are at least as far apart as a population on that range can be.
    The sample standard deviation of a small sample can exceed the population
    bound — two repeats at opposite ends give ``0.707`` on a unit range — so
    the result is clamped rather than allowed to go negative.

    Standard deviation rather than variance because it shares the units of the
    scores: halving the spread of the repeats halves the distance from a
    perfect reading. Fewer than two scores carries no information about
    spread, so that reports ``1.0``.
    """
    if len(scores) < 2:
        return 1.0

    low, high = score_range
    half_width = (high - low) / 2
    if half_width <= 0:
        raise ValueError(f"'score_range' must be non-empty, got {score_range}")

    normalized = 1.0 - statistics.stdev(scores) / half_width
    return min(1.0, max(0.0, normalized))


def decision_flip_rate(successes: Sequence[bool]) -> Optional[float]:
    """Fraction of replicate *pairs* that landed on opposite sides of the
    threshold.

    With ``k`` replicates of which ``p`` passed, the disagreeing pairs number
    ``p * (k - p)`` out of ``k * (k - 1) / 2`` total pairs. ``0.0`` means every
    repeat reached the same verdict, and the value rises as the split
    approaches even.

    This is the unbiased estimator of ``2q(1-q)``, the probability that two
    independent repeats disagree when the judge passes with probability ``q``.
    That probability is bounded by ``0.5``, but the unbiased estimator is not:
    an even split at small ``k`` reports above it (``1.0`` at ``k=2``, ``0.667``
    at ``k=4``). Read the number as "how many of my pairs disagreed", and only
    read it as a probability estimate once ``k`` is reasonably large.

    Returns ``None`` when there are fewer than two replicates, or when the
    judge has no threshold and therefore produces no pass/fail decision.
    """
    k = len(successes)
    if k < 2:
        return None

    passed = sum(1 for success in successes if success)
    disagreeing_pairs = passed * (k - passed)
    total_pairs = k * (k - 1) / 2
    return disagreeing_pairs / total_pairs


def _percentile(sorted_values: List[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"'fraction' must be in [0, 1], got {fraction}")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower = sorted_values[lower_index]
    return lower + weight * (sorted_values[upper_index] - lower)


def bootstrap_mean_interval(
    scores: Sequence[float],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap confidence interval for the mean of ``scores``.

    Resamples ``scores`` with replacement ``resamples`` times, takes the mean
    of each resample, and reads the interval off the percentiles of that
    distribution. This says how precisely the repeats pin down the judge's
    central score — a wide interval on a handful of replicates is a signal to
    raise the replicate count, not a finding about the judge.

    ``seed`` makes the interval reproducible; pass ``None`` for a fresh draw
    each call. Returns ``None`` for an empty input.
    """
    if resamples < 1:
        raise ValueError(f"'resamples' must be at least 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"'confidence' must be strictly between 0 and 1, got {confidence}"
        )
    if not scores:
        return None
    if len(scores) == 1:
        return (scores[0], scores[0])

    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += scores[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    return (_percentile(means, tail), _percentile(means, 1.0 - tail))
