from typing import Protocol, Sequence
import statistics


class Aggregator(Protocol):
    def __call__(self, scores: Sequence[float]) -> float: ...


def mean_of_all(scores: Sequence[float]) -> float:
    return statistics.fmean(scores) if scores else 0.0


def median_of_all(scores: Sequence[float]) -> float:
    return statistics.median(scores) if scores else 0.0
