from typing import Optional, List, Union, Dict, Any, Iterator, Mapping
from dataclasses import dataclass, field
from collections import Counter
from pydantic import BaseModel

from deepeval.test_run.api import MetricData, TurnApi
from deepeval.test_case import MLLMImage
from deepeval.test_run import TestRun


@dataclass
class TestResult:
    """Returned from run_test"""

    __test__ = False
    name: str
    success: bool
    metrics_data: Union[List[MetricData], None]
    conversational: bool
    index: Optional[int] = None
    multimodal: Optional[bool] = None
    input: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    actual_output: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    turns: Optional[List[TurnApi]] = None
    metadata: Optional[Dict] = None


class EvaluationResult(BaseModel):
    test_results: List[TestResult]
    confident_link: Optional[str]
    test_run_id: Optional[str]


class PostExperimentRequest(BaseModel):
    testRuns: List[TestRun]
    name: Optional[str]


@dataclass
class ArenaResult(Mapping[str, int]):
    """Structured return value for :func:`deepeval.evaluate.compare`.

    Implements the ``Mapping[str, int]`` protocol so legacy code like
    ``result["GPT-4"]`` or ``dict(result)`` keeps working — the mapping
    behaviour is exactly the old ``winner_counts`` dictionary. New code
    should prefer the explicit attributes below and the
    :meth:`analyze` helper for significance testing.
    """

    __test__ = False

    # ------------------------------------------------------------------
    # Per-case winners — the compare() function used to throw these away
    # after aggregating counts. Keeping them is the single biggest enabler:
    # downstream code (significance tests, per-round reporting, drift
    # detection) no longer has to re-run or patch the pipeline to get them.
    # ------------------------------------------------------------------
    winners: List[Optional[str]] = field(default_factory=list)
    """Per-case winner name (or ``None`` for a tie), in input order."""

    # ------------------------------------------------------------------
    # Raw counts — matches the legacy Dict[str, int] payload.
    # Stored as a Counter so `.most_common()` / `.elements()` work.
    # ------------------------------------------------------------------
    _counts: Counter = field(default_factory=Counter)

    # ------------------------------------------------------------------
    # Convenience: contestant names observed across all input test cases.
    # ------------------------------------------------------------------
    contestants: List[str] = field(default_factory=list)
    """Unique contestant names, in order of first appearance."""

    # ------------------------------------------------------------------
    # Metadata surfaced by wrap_up_experiment (useful for reports /
    # programmatic consumers without having to capture stdout).
    # ------------------------------------------------------------------
    n_cases: int = 0
    run_duration: float = 0.0

    # ---- Mapping[str, int] backward-compat layer --------------------
    # Old code does:
    #     counts = compare(...)
    #     counts["GPT-4"]         # __getitem__
    #     for name in counts: ... # __iter__
    #     len(counts)             # __len__
    #     dict(counts)            # works via Mapping mixin
    # We implement exactly these four so the behaviour is indistinguishable
    # from the historical Dict[str, int] return value.

    def __getitem__(self, name: str) -> int:  # type: ignore[override]
        value = self._counts.get(name)
        if value is None:
            raise KeyError(name)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)

    # ---- Dict-style helpers (users expect these on dict-like things) --
    def get(self, name: str, default: int = 0) -> int:
        return self._counts.get(name, default)

    def keys(self):
        return self._counts.keys()

    def values(self):
        return self._counts.values()

    def items(self):
        return self._counts.items()

    def to_dict(self) -> Dict[str, int]:
        """Explicit conversion to the legacy ``{contestant: wins}`` dict."""
        return dict(self._counts)

    # ---- Integrated significance analysis ---------------------------
    def analyze(
        self,
        model_a: Optional[str] = None,
        model_b: Optional[str] = None,
        alpha: float = 0.05,
        tie_strategy: str = "drop",
    ):
        """Run a significance test on a head-to-head pair.

        ``model_a`` / ``model_b`` are inferred automatically for two-model
        arenas; three-or-more-way comparisons require an explicit pair.
        """
        # Lazy import — avoids a circular import: types.py is imported by
        # compare.py which is imported by analyze.py's package __init__.
        from deepeval.evaluate.analyze import (
            analyze_compare_results,
            _TIE,
        )

        pair = [c for c in self.contestants]
        if model_a is None and model_b is None:
            if len(pair) == 2:
                model_a, model_b = pair[0], pair[1]
            else:
                raise ValueError(
                    f"Arena has {len(pair)} contestants "
                    f"({pair!r}); pass model_a and model_b explicitly to "
                    "analyze a specific pair."
                )
        elif model_a is None or model_b is None:
            raise ValueError(
                "Pass both model_a and model_b, or neither for a 2-way arena."
            )

        # mypy won't know the run-time literal check, so cast locally.
        strategy: Any = tie_strategy
        return analyze_compare_results(
            winners=self.winners,
            model_a=model_a,
            model_b=model_b,
            alpha=alpha,
            tie_strategy=strategy,
        )
