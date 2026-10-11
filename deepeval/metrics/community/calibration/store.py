"""Per-run row store for the calibration metric.

DeepEval's ``BaseMetric.measure()`` / ``a_measure()`` see ONE test case at a
time and there is no "run finished" signal, so per-row (confidence, outcome)
pairs have to be accumulated somewhere shared. Under ``evaluate(run_async=True)``
DeepEval re-instantiates the metric per test case (``copy_metrics``) and only
forwards ``__init__`` params by reference, so the metric receives this store
as a constructor argument (same attribute name) and every copy shares it.

Design (see decisions-and-constraints.md, "Concurrency model"):
- No module-level state: every run gets its own store from
  ``create_calibration_store()``.
- ``threading.Lock`` guards mutation. The critical section is a dict write /
  list copy and is never held across an ``await`` or while user code runs.
- Aggregation is lazy and idempotent: ``aggregate(fn)`` snapshots the rows
  under the lock, then runs ``fn`` OUTSIDE the lock. Any call sees a
  consistent set of rows-so-far; the last call to finish sees the full run.
- Row identity: ``key=None`` always appends; a non-None key overwrites, so a
  retried/duplicated test case with a stable key is not double counted.
"""

import math
import threading
from typing import Callable, Dict, Hashable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class _AnonKey:
    """Unique key for rows added without an identity (always appended)."""

    __slots__ = ()


class CalibrationRunStore:
    """Thread-safe accumulator of (confidence, outcome) rows for one run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: Dict[Hashable, Tuple[float, int]] = {}

    @staticmethod
    def _validate(confidence: float, outcome: object) -> Tuple[float, int]:
        # Same contract as the math core, enforced at the door so a bad row
        # can never poison every later aggregate.
        if isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise ValueError(f"confidence must be a number, got {confidence!r}")
        c = float(confidence)
        if math.isnan(c) or not (0.0 <= c <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")
        if isinstance(outcome, bool):
            o = int(outcome)
        elif isinstance(outcome, int) and outcome in (0, 1):
            o = outcome
        else:
            raise ValueError(f"outcome must be 0/1/bool, got {outcome!r}")
        return c, o

    def add(
        self,
        confidence: float,
        outcome: object,
        key: Optional[Hashable] = None,
    ) -> int:
        """Record one row; return the number of rows now stored.

        ``key=None``: always appended. ``key`` given: overwrites any earlier
        row with the same key (retry-safe), keeping its original position.
        """
        row = self._validate(confidence, outcome)
        with self._lock:
            self._rows[_AnonKey() if key is None else key] = row
            return len(self._rows)

    def snapshot(self) -> Tuple[List[float], List[int]]:
        """Consistent copy of rows-so-far as (confidences, outcomes)."""
        with self._lock:
            rows = list(self._rows.values())
        return [c for c, _ in rows], [o for _, o in rows]

    def aggregate(
        self, fn: Callable[[List[float], List[int]], T]
    ) -> T:
        """Run ``fn(confidences, outcomes)`` on a snapshot, outside the lock.

        Idempotent: safe to call after every row. Propagates whatever ``fn``
        raises (the math core raises ValueError on an empty set).
        """
        confidences, outcomes = self.snapshot()
        return fn(confidences, outcomes)

    def reset(self) -> None:
        """Drop all rows (explicit run boundary for a reused store)."""
        with self._lock:
            self._rows.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


def create_calibration_store() -> CalibrationRunStore:
    """Factory: a fresh, empty, independent store for one evaluation run."""
    return CalibrationRunStore()
