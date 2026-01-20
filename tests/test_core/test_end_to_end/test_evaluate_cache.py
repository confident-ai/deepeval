import os

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.test_run.test_run import LATEST_TEST_RUN_FILE_PATH
from deepeval.test_run.cache import CACHE_FILE_NAME

from .helpers import (
    build_single_turn_dataset,
    build_llm_test_cases_from_goldens,
    DeterministicPassingMetric,
)


# Module-level counter for tracking metric executions
_metric_call_counter = {"calls": 0}


def _reset_metric_counter():
    """Reset the metric call counter to zero."""
    _metric_call_counter["calls"] = 0


class CountingMetric(BaseMetric):
    """
    A deterministic metric that counts how many times it is executed.
    Used to verify cache hit/miss behavior.
    """

    _required_params = [LLMTestCaseParams.ACTUAL_OUTPUT]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "CountingMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        _metric_call_counter["calls"] += 1
        self.score = 1.0
        self.reason = "Counting metric always passes"
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return True


def test_write_cache_creates_artifacts_on_disk():
    """
    Verify that write_cache=True creates cache files on disk.

    When evaluate() runs with write_cache=True, it should persist:
    - Metric cache to CACHE_FILE_NAME (.deepeval-cache.json)
    - Latest test run data to LATEST_TEST_RUN_FILE_PATH (.latest_test_run.json)
    """
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=True, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # Cache artifacts should exist after evaluation
    assert os.path.isfile(CACHE_FILE_NAME), (
        f"Expected CACHE_FILE_NAME ({CACHE_FILE_NAME}) to exist after "
        "evaluate() with write_cache=True"
    )
    assert os.path.isfile(LATEST_TEST_RUN_FILE_PATH), (
        f"Expected LATEST_TEST_RUN_FILE_PATH ({LATEST_TEST_RUN_FILE_PATH}) "
        "to exist after evaluate() with write_cache=True"
    )


def test_use_cache_true_reuses_cached_results():
    """
    Verify that use_cache=True reuses cached metric results.

    When evaluate() runs twice with identical inputs and use_cache=True,
    the second run should use cached results and NOT re-execute the metric.
    """
    _reset_metric_counter()

    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)
    num_test_cases = len(test_cases)

    # First run: metrics should execute
    evaluate(
        test_cases=test_cases,
        metrics=[CountingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=True, use_cache=True),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    calls_after_first_run = _metric_call_counter["calls"]
    assert calls_after_first_run == num_test_cases, (
        f"Expected {num_test_cases} metric calls after first run, "
        f"got {calls_after_first_run}"
    )

    # Second run with identical inputs: should use cache, no new metric calls
    evaluate(
        test_cases=test_cases,
        metrics=[CountingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=True, use_cache=True),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    calls_after_second_run = _metric_call_counter["calls"]
    assert calls_after_second_run == calls_after_first_run, (
        f"Expected no additional metric calls on second run with use_cache=True. "
        f"Calls after first run: {calls_after_first_run}, "
        f"calls after second run: {calls_after_second_run}"
    )


def test_use_cache_false_recomputes_metrics():
    """
    Verify that use_cache=False recomputes metrics even when cache exists.

    When evaluate() runs twice with identical inputs but use_cache=False,
    metrics should be re-executed on the second run.
    """
    _reset_metric_counter()

    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)
    num_test_cases = len(test_cases)

    # First run: metrics should execute
    evaluate(
        test_cases=test_cases,
        metrics=[CountingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=True, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    calls_after_first_run = _metric_call_counter["calls"]
    assert calls_after_first_run == num_test_cases, (
        f"Expected {num_test_cases} metric calls after first run, "
        f"got {calls_after_first_run}"
    )

    # Second run with use_cache=False: should recompute, more metric calls
    evaluate(
        test_cases=test_cases,
        metrics=[CountingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=True, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    calls_after_second_run = _metric_call_counter["calls"]
    expected_calls = num_test_cases * 2
    assert calls_after_second_run == expected_calls, (
        f"Expected {expected_calls} total metric calls after second run "
        f"with use_cache=False (recomputation), got {calls_after_second_run}"
    )
