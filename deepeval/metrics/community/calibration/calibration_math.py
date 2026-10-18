"""Pure calibration math: Expected Calibration Error (ECE) and Brier score.

Zero third-party and zero DeepEval imports (stdlib only) so it can be unit-tested
in isolation and later dropped into deepeval/metrics/community/calibration/.

Input contract (source-agnostic, see decisions-and-constraints.md):
  confidences: probabilities in [0, 1] that each prediction is correct.
               Origin is irrelevant (logprob, retrieval distance, classifier...).
  corrects:    1/0 or True/False -- did the *event the confidence is a
               probability of* occur?  This function assigns ONE meaning to
               this flag; deciding what "correct" means for a given row
               (retrieval hit vs. appropriate abstention) is the caller's job.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Sequence, Tuple, Union

Number = Union[int, float]
Outcome = Union[bool, int]


class BinStat(NamedTuple):
    lower: float  # inclusive (last bin: upper also inclusive)
    upper: float
    count: int
    avg_confidence: float  # 0.0 when count == 0
    accuracy: float  # 0.0 when count == 0


def _validate(
    confidences: Sequence[Number], corrects: Sequence[Outcome], n_bins: int
) -> None:
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError(f"n_bins must be an int >= 1, got {n_bins!r}")
    if len(confidences) != len(corrects):
        raise ValueError(
            f"confidences and corrects must be the same length "
            f"({len(confidences)} != {len(corrects)})"
        )
    if len(confidences) == 0:
        raise ValueError("need at least one (confidence, correct) pair")
    for i, c in enumerate(confidences):
        if (
            isinstance(c, bool)
            or not isinstance(c, (int, float))
            or math.isnan(c)
        ):
            raise ValueError(
                f"confidences[{i}] must be a real number, got {c!r}"
            )
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidences[{i}]={c!r} outside [0, 1]")
    for i, o in enumerate(corrects):
        if o not in (0, 1):  # True/False/0/1 (and 0.0/1.0) only
            raise ValueError(f"corrects[{i}] must be a bool or 0/1, got {o!r}")


def _bin_index(confidence: float, n_bins: int) -> int:
    """Equal-width bins [0,1/n), [1/n,2/n), ..., [(n-1)/n, 1] (last bin closed).

    int(c * n) alone is NOT safe: e.g. (31/39) * 39 == 30.999999999999996, which
    would drop a value sitting exactly on a bin edge into the wrong bin. The
    nudges below compare against the same float edges (k / n) that
    calibration_bins() reports, so membership is always consistent with them.
    """
    idx = min(int(confidence * n_bins), n_bins - 1)
    if idx + 1 < n_bins and confidence >= (idx + 1) / n_bins:
        idx += 1
    elif idx > 0 and confidence < idx / n_bins:
        idx -= 1
    return idx


def calibration_bins(
    confidences: Sequence[Number], corrects: Sequence[Outcome], n_bins: int = 10
) -> List[BinStat]:
    """Per-bin stats (also the data behind a reliability diagram)."""
    _validate(confidences, corrects, n_bins)
    counts = [0] * n_bins
    conf_sum = [0.0] * n_bins
    hit_sum = [0.0] * n_bins
    for c, o in zip(confidences, corrects):
        b = _bin_index(float(c), n_bins)
        counts[b] += 1
        conf_sum[b] += float(c)
        hit_sum[b] += float(o)
    return [
        BinStat(
            lower=b / n_bins,
            upper=(b + 1) / n_bins,
            count=counts[b],
            avg_confidence=conf_sum[b] / counts[b] if counts[b] else 0.0,
            accuracy=hit_sum[b] / counts[b] if counts[b] else 0.0,
        )
        for b in range(n_bins)
    ]


def calculate_calibration(
    confidences: Sequence[Number], corrects: Sequence[Outcome], n_bins: int = 10
) -> Tuple[float, float]:
    """Return (ece, brier_score).

    ECE   = sum_b (n_b / N) * |accuracy_b - avg_confidence_b|   (equal-width bins,
            Guo et al. 2017; empty bins contribute 0)
    Brier = (1 / N) * sum_i (confidence_i - correct_i)^2        (binary form)
    Both lie in [0, 1]; lower is better.
    """
    bins = calibration_bins(confidences, corrects, n_bins)
    n = len(confidences)
    ece = sum(
        (b.count / n) * abs(b.accuracy - b.avg_confidence)
        for b in bins
        if b.count
    )
    brier = (
        sum((float(c) - float(o)) ** 2 for c, o in zip(confidences, corrects))
        / n
    )
    return ece, brier
