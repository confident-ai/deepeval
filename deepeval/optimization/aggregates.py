# deepeval/optimization/aggregates.py
from typing import Protocol, List
import statistics


class Aggregator(Protocol):
    """Aggregates a list of instance-level scores into a single scalar."""

    def __call__(self, instance_scores: List[float]) -> float: ...


def mean_of_all(instance_scores: List[float]) -> float:
    """Arithmetic mean of all instance scores (0.0 if empty)."""
    return 0.0 if not instance_scores else statistics.fmean(instance_scores)


def median_of_all(instance_scores: List[float]) -> float:
    """Median of all instance scores (0.0 if empty)."""
    return 0.0 if not instance_scores else statistics.median(instance_scores)


def top_k_mean_of_best(number_of_top_values: int):
    """
    Mean of the top-K instance scores. K is clamped to [1, len(scores)].
    """

    def _aggregate(instance_scores: List[float]) -> float:
        if not instance_scores:
            return 0.0
        k = max(1, min(number_of_top_values, len(instance_scores)))
        best_k = sorted(instance_scores, reverse=True)[:k]
        return statistics.fmean(best_k)

    return _aggregate


def trimmed_mean_symmetric(trim_fraction: float):
    """
    Symmetric trimmed mean: drops the lowest/highest trim_fraction of scores.
    trim_fraction must be in [0.0, 0.5). Falls back to mean if the trim removes nothing.
    """
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0.0, 0.5).")

    def _aggregate(instance_scores: List[float]) -> float:
        if not instance_scores:
            return 0.0
        sorted_scores = sorted(instance_scores)
        n = len(sorted_scores)
        cut = min(int(n * trim_fraction), (n - 1) // 2)
        core = sorted_scores[cut : n - cut] if cut else sorted_scores
        return statistics.fmean(core)

    return _aggregate
