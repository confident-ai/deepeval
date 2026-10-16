"""Dataset calibration statistics, computed using only the standard library."""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class CalibrationResult:
    """Dataset ECE and binary Brier score. Both are lower-is-better."""

    ece: float
    brier: float


def calculate_calibration(
    confidences: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> CalibrationResult:
    """Compare reported confidence with empirical binary correctness.

    Each confidence must be a finite real number in [0, 1], paired with a
    numeric outcome equal to 0 or 1. Booleans and non-real values are rejected.
    Inputs must be non-empty sequences of equal length; n_bins must be a
    positive integer (not a boolean).

    ECE uses equal-width bins [i/n_bins, (i+1)/n_bins), with 1.0 included
    in the final bin. Empty bins contribute zero. Brier score is the mean
    squared difference between confidence and outcome, independent of bins.

    Raises TypeError for invalid element/bin-count types and ValueError for
    invalid values, empty inputs, or mismatched lengths. This function has
    no run state and makes no model calls. Its result describes the supplied
    dataset and cannot be passed as a metric to evaluate().
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral):
        raise TypeError("n_bins must be a positive integer")
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")
    if len(confidences) == 0:
        raise ValueError("confidences and outcomes must be non-empty")

    for index, (confidence, outcome) in enumerate(zip(confidences, outcomes)):
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise TypeError(f"confidences[{index}] must be a real number")
        if not 0 <= confidence <= 1 or not isfinite(confidence):
            raise ValueError(
                f"confidences[{index}] must be finite and between 0 and 1"
            )
        if isinstance(outcome, bool) or not isinstance(outcome, Real):
            raise TypeError(f"outcomes[{index}] must be a real number")
        if outcome not in (0, 1):
            raise ValueError(f"outcomes[{index}] must be 0 or 1")

    # Compare against explicit edges: int(confidence * n_bins) can round
    # down at a boundary (e.g. 31/39). Interior edges also keep 1.0 in range.
    edges = [index / n_bins for index in range(1, n_bins)]
    bin_confidences = [[] for _ in range(n_bins)]
    bin_correct = [0] * n_bins
    for confidence, outcome in zip(confidences, outcomes):
        index = bisect_right(edges, confidence)
        bin_confidences[index].append(confidence)
        bin_correct[index] += int(outcome)

    total = len(confidences)
    ece = fsum(
        (len(values) / total)
        * abs(fsum(values) / len(values) - correct / len(values))
        for values, correct in zip(bin_confidences, bin_correct)
        if values
    )
    # Normalize validated real numbers before subtraction, since unsigned
    # numeric scalar types can underflow when the outcome exceeds confidence.
    brier = (
        fsum(
            (float(confidence) - float(outcome)) ** 2
            for confidence, outcome in zip(confidences, outcomes)
        )
        / total
    )
    return CalibrationResult(ece=ece, brier=brier)
