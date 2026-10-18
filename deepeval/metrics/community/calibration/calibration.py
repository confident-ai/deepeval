"""CalibrationMetric (ECE + Brier) -- DRAFT, Phase 5.

Goes to deepeval/metrics/community/calibration/calibration.py.
Assumes Phase 2-4 modules in the same package (names per decisions file):
    calibration_math: calculate_calibration, calibration_bins, BinStat
    store:            CalibrationRunStore, create_calibration_store
    schema:           CalibrationResult, CalibrationBin
VERIFY against the clone: BinStat attribute names, calibration_bins() args.
"""

import os
from typing import Any, List, Optional, Tuple

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from .calibration_math import calculate_calibration, calibration_bins
from .schema import CalibrationBin, CalibrationResult
from .store import CalibrationRunStore, create_calibration_store

OUTCOME_DEFINITION = (
    "1 if the system's answer was correct, else 0. A query with no "
    "ground-truth answer counts as 0 (any answer would be wrong). "
    "Appropriateness of abstention is NOT scored."
)

# Below this many rows, ECE/Brier are too noisy to trust.
SMALL_SAMPLE_WARNING_THRESHOLD = 30

# Upper bound on n_bins: bin arrays are allocated up front, so an absurd
# value costs seconds of work and memory before any row is read.
MAX_N_BINS = 1000


def _coerce_row(confidence: Any, outcome: Any) -> Tuple[float, int]:
    """Cast metadata values to what the store accepts (float, int 0/1).

    numpy scalars are accepted here (the store rejects them by design).
    bool confidences and strings are rejected: almost always a wrong key.
    """
    if isinstance(confidence, (bool, str)) or confidence is None:
        raise ValueError(
            f"confidence must be a number in [0, 1], got {confidence!r}"
        )
    if isinstance(outcome, str) or outcome is None:
        raise ValueError(f"outcome must be True/False/0/1, got {outcome!r}")
    try:
        conf = float(confidence)
        out = float(outcome)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"non-numeric calibration row: confidence={confidence!r}, "
            f"outcome={outcome!r}"
        )
    if out not in (0.0, 1.0):
        raise ValueError(f"outcome must be True/False/0/1, got {outcome!r}")
    return conf, int(out)


class CalibrationMetric(BaseMetric):
    """Measures whether confidence scores are trustworthy in aggregate.

    Each test case supplies a confidence and whether the system's answer was
    correct (both via ``test_case.metadata``). Rows accumulate in a run store;
    every ``measure()`` recomputes ECE and Brier over the rows seen so far.

    Per-case ``score`` = 1 - (confidence - outcome)^2 (per-row Brier
    complement). Aggregates live in ``score_breakdown`` and
    ``get_calibration_result()``.

    Notes
    -----
    * No LLM, no cost, no ``model``.
    * Confidence can come from anywhere (logprobs, retrieval distance, a
      classifier); it is not derived from logprobs.
    * Single process only: under ``deepeval test run -n N`` each xdist worker
      has its own store.
    * In async ``evaluate()`` DeepEval copies the metric per test case, so read
      the aggregate from ``your_metric.get_calibration_result()``, not from
      per-case metric objects.
    * Set ``name`` on test cases to make reruns idempotent.
    """

    # Metadata-only: INPUT is a mandatory LLMTestCase field anyway.
    _required_params: List[SingleTurnParams] = [SingleTurnParams.METADATA]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        confidence_key: str = "confidence",
        correct_key: str = "correct",
        n_bins: int = 10,
        run_store: Optional[CalibrationRunStore] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if isinstance(n_bins, bool) or not isinstance(n_bins, int):
            raise ValueError("n_bins must be an int")
        if not 1 <= n_bins <= MAX_N_BINS:
            raise ValueError(f"n_bins must be in [1, {MAX_N_BINS}]")
        # copy_metrics() re-instantiates from vars(self) filtered by __init__
        # param names: every param MUST be stored under the same attr name,
        # and the store MUST be `run_store` or it is silently dropped.
        self.threshold = 1 if strict_mode else threshold
        self.confidence_key = confidence_key  # NOT `confidence` (BaseMetric)
        self.correct_key = correct_key
        self.n_bins = n_bins
        self.run_store = (
            run_store if run_store is not None else create_calibration_store()
        )
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        # Deterministic metric: no judge model.
        self.model = None
        self.using_native_model = True
        self.evaluation_model = None

    # -- public API --------------------------------------------------------

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            self._calculate_metric(test_case)
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # No I/O to await; the store lock is never held across an await.
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self._calculate_metric(test_case)
            return self.score

    def get_calibration_result(self) -> CalibrationResult:
        """Aggregate over all rows in the run store so far (idempotent)."""
        return self.run_store.aggregate(self._build_result)

    def reset(self) -> None:
        """Start a new run on this store (clears all rows)."""
        self.run_store.reset()

    def is_successful(self) -> bool:
        # Per-case: score >= threshold (BaseMetric default). The aggregate
        # ECE gate is exposed via get_calibration_result(), not here.
        return super().is_successful()

    @property
    def __name__(self):
        return "Calibration (ECE / Brier)"

    # -- internals ---------------------------------------------------------

    def _calculate_metric(self, test_case: LLMTestCase) -> None:
        self.error = None
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )
        metadata = test_case.metadata
        missing = [
            f"'{k}'"
            for k in (self.confidence_key, self.correct_key)
            if k not in metadata
        ]
        if missing:
            msg = (
                f"{', '.join(missing)} missing from test_case.metadata for "
                f"the '{self.__name__}' metric"
            )
            self.error = msg
            raise MissingTestCaseParamsError(msg)

        try:
            conf, outcome = _coerce_row(
                metadata[self.confidence_key], metadata[self.correct_key]
            )
            self.run_store.add(conf, outcome, key=test_case.name)
        except (TypeError, ValueError) as e:
            self.error = str(e)
            raise

        result = self.get_calibration_result()
        self.score = 1.0 - (conf - outcome) ** 2
        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0
        self.score_breakdown = {
            "ece": result.ece,
            "brier_score": result.brier_score,
            "n_samples": result.n_samples,
            "n_bins": result.n_bins,
            "confidence": conf,
            "outcome": outcome,
        }
        self.success = self.is_successful()
        self.reason = result.reasoning if self.include_reason else None
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Row: confidence={conf}, outcome={outcome}",
                f"Rows so far: {result.n_samples}",
                f"ECE: {result.ece:.4f}, Brier: {result.brier_score:.4f}",
                f"Per-case score: {self.score}",
                f"Reason: {self.reason}",
            ],
        )

    def _build_result(
        self, confidences: List[float], outcomes: List[int]
    ) -> CalibrationResult:
        ece, brier = calculate_calibration(confidences, outcomes, self.n_bins)
        stats = calibration_bins(confidences, outcomes, self.n_bins)
        bins = [
            CalibrationBin(
                lower=b.lower,
                upper=b.upper,
                count=b.count,
                avg_confidence=b.avg_confidence,
                accuracy=b.accuracy,
            )
            for b in stats
        ]
        n = len(confidences)
        warnings: List[str] = []
        if n < SMALL_SAMPLE_WARNING_THRESHOLD:
            warnings.append(
                f"Only {n} rows (< {SMALL_SAMPLE_WARNING_THRESHOLD}); ECE and "
                "Brier are noisy at this sample size."
            )
        if os.environ.get("PYTEST_XDIST_WORKER"):
            warnings.append(
                "Running under pytest-xdist: this aggregate covers only "
                "this worker's rows. Run calibration single-process."
            )
        return CalibrationResult(
            ece=ece,
            brier_score=brier,
            n_bins=self.n_bins,
            n_samples=n,
            outcome_definition=OUTCOME_DEFINITION,
            bins=bins,
            warnings=warnings,
            reasoning=self._reasoning(ece, brier, n, bins, warnings),
        )

    @staticmethod
    def _reasoning(
        ece: float,
        brier: float,
        n: int,
        bins: List[CalibrationBin],
        warnings: List[str],
    ) -> str:
        parts = [f"ECE={ece:.4f}, Brier={brier:.4f} over {n} rows."]
        for b in bins:
            if b.count:
                parts.append(
                    f"[{b.lower:.2f}, {b.upper:.2f}): n={b.count}, "
                    f"avg_conf={b.avg_confidence:.3f}, acc={b.accuracy:.3f}"
                )
        parts.extend(f"Warning: {w}" for w in warnings)
        return " ".join(parts)
