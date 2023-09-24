"""Function for running test
"""
import os
from typing import List, Optional, Union
import time
from dataclasses import dataclass
from .retry import retry
from .client import Client
from .constants import (
    IMPLEMENTATION_ID_ENV,
    LOG_TO_SERVER_ENV,
    PYTEST_RUN_ENV_VAR,
)
from .get_api_key import _get_api_key, _get_implementation_name
from .metrics import Metric
from .test_case import LLMTestCase, TestCase, SearchTestCase
from .api import TestRun


def _is_api_key_set():
    result = _get_api_key()
    # if result == "" or result is None:
    #     warnings.warn(
    #         """API key is not set - you won't be able to log to the DeepEval dashboard. Please set it by running `deepeval login`"""
    #     )
    if result == "" or result is None:
        return False
    return True


def _is_send_okay():
    # DOing this until the API endpoint is fixed
    return _is_api_key_set() and os.getenv(LOG_TO_SERVER_ENV) != "Y"


def _get_init_values(metric: Metric):
    # We use this method for sending useful metadata
    init_values = {
        param: getattr(metric, param)
        for param in vars(metric)
        if isinstance(getattr(metric, param), (str, int, float))
    }
    return init_values


@dataclass
class TestResult:
    """Returned from run_test"""

    success: bool
    score: float
    metric_name: str
    query: str
    output: str
    expected_output: str
    metadata: Optional[dict]
    context: str

    def __gt__(self, other: "TestResult") -> bool:
        """Greater than comparison based on score"""
        return self.score > other.score

    def __lt__(self, other: "TestResult") -> bool:
        """Less than comparison based on score"""
        return self.score < other.score


def run_test(
    test_cases: Union[TestCase, LLMTestCase, SearchTestCase, List[LLMTestCase]],
    metrics: List[Metric],
    max_retries: int = 1,
    delay: int = 1,
    min_success: int = 1,
    raise_error: bool = False,
) -> List[TestResult]:
    """
    Args:
        test_cases: Either a single test case or a list of test cases to run
        metrics: List of metrics to run
        raise_error: Whether to raise an error if a metric fails
        max_retries: Maximum number of retries for each metric measurement
        delay: Delay in seconds between retries
        min_success: Minimum number of successful measurements required

    Example:
        >>> from deepeval.metrics.facutal_consistency import FactualConsistencyMetric
        >>> from deepeval.test_case import LLMTestCase
        >>> from deepeval.run_test import run_test
        >>> metric = FactualConsistencyMetric()
        >>> test_case = LLMTestCase(
        ...     query="What is the capital of France?",
        ...     output="Paris",
        ...     expected_output="Paris",
        ...     context="Geography",
        ... )
        >>> run_test(test_case, metric)
    """
    if isinstance(test_cases, TestCase):
        test_cases = [test_cases]

    test_results = []
    for test_case in test_cases:
        for metric in metrics:
            test_start_time = time.perf_counter()

            @retry(
                max_retries=max_retries, delay=delay, min_success=min_success
            )
            def measure_metric():
                score = metric.measure(test_case)
                success = metric.is_successful()
                if isinstance(test_case, LLMTestCase):
                    test_result = TestResult(
                        success=success,
                        score=score,
                        metric_name=metric.__name__,
                        query=test_case.query if test_case.query else "-",
                        output=test_case.output if test_case.output else "-",
                        expected_output=test_case.expected_output
                        if test_case.expected_output
                        else "-",
                        metadata=None,
                        context=test_case.context,
                    )
                elif isinstance(test_case, SearchTestCase):
                    test_result = TestResult(
                        success=success,
                        score=score,
                        metric_name=metric.__name__,
                        query=test_case.query if test_case.query else "-",
                        output=test_case.output_list
                        if test_case.output_list
                        else "-",
                        expected_output=test_case.golden_list
                        if test_case.golden_list
                        else "-",
                        metadata=None,
                        context="-",
                    )
                else:
                    raise ValueError("TestCase not supported yet.")
                test_results.append(test_result)

                # Load the test_run and add the test_case regardless of the success of the test
                test_end_time = time.perf_counter()
                run_duration = test_end_time - test_start_time
                if os.getenv(PYTEST_RUN_ENV_VAR):
                    test_run = TestRun.load()
                    metric.score = score
                    test_run.add_llm_test_case(
                        test_case=test_case,
                        metrics=[metric],
                        run_duration=run_duration,
                    )
                    test_run.save()

                if raise_error:
                    assert (
                        metric.is_successful()
                    ), f"{metric.__name__} failed. Score: {score}."

            measure_metric()

    return test_results


def assert_test(
    test_cases: Union[LLMTestCase, List[LLMTestCase]],
    metrics: List[Metric],
    max_retries: int = 1,
    delay: int = 1,
    min_success: int = 1,
) -> List[TestResult]:
    """Assert a test"""
    return run_test(
        test_cases=test_cases,
        metrics=metrics,
        max_retries=max_retries,
        delay=delay,
        min_success=min_success,
        raise_error=True,
    )
