"""Function for running test
"""
import os
import warnings
from typing import List, Optional, Union
import time
from dataclasses import dataclass
from .retry import retry
from .constants import (
    LOG_TO_SERVER_ENV,
    PYTEST_RUN_ENV_VAR,
)
from .get_api_key import _get_api_key
from .metrics import BaseMetric
from .test_case import LLMTestCase, TestCase
from deepeval.test_run import test_run_manager, TestRun


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


def _get_init_values(metric: BaseMetric):
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

    def __post_init__(self):
        """Ensures score is between 0 and 1 after initialization"""
        original_score = self.score
        self.score = min(max(0, self.score), 1)
        if self.score != original_score:
            warnings.warn(
                "The score was adjusted to be within the range [0, 1]."
            )

    def __gt__(self, other: "TestResult") -> bool:
        """Greater than comparison based on score"""
        return self.score > other.score

    def __lt__(self, other: "TestResult") -> bool:
        """Less than comparison based on score"""
        return self.score < other.score


def create_test_result(
    test_case: LLMTestCase,
    success: bool,
    score: float,
    metric: float,
) -> TestResult:
    if isinstance(test_case, LLMTestCase):
        return TestResult(
            success=success,
            score=score,
            metric_name=metric.__name__,
            query=test_case.input if test_case.input else "-",
            output=test_case.actual_output if test_case.actual_output else "-",
            expected_output=test_case.expected_output
            if test_case.expected_output
            else "-",
            metadata=None,
            context=test_case.context,
        )
    else:
        raise ValueError("TestCase not supported yet.")


def run_test(
    test_cases: Union[TestCase, LLMTestCase, List[LLMTestCase]],
    metrics: List[BaseMetric],
    max_retries: int = 1,
    delay: int = 1,
    min_success: int = 1,
    raise_error: bool = False,
) -> List[TestResult]:
    if isinstance(test_cases, TestCase):
        test_cases = [test_cases]

    test_results = []
    test_run = test_run_manager.get_test_run()
    for test_case in test_cases:
        failed_metrics = []
        for metric in metrics:
            test_start_time = time.perf_counter()

            # @retry(
            #     max_retries=max_retries,
            #     delay=delay,
            #     min_success=min_success,
            # )
            # def measure_metric():
            score = metric.measure(test_case)
            success = metric.is_successful()
            test_result = create_test_result(test_case, success, score, metric)
            test_results.append(test_result)

            # Load the test_run and add the test_case regardless of the success of the test
            test_end_time = time.perf_counter()
            run_duration = test_end_time - test_start_time
            if os.getenv(PYTEST_RUN_ENV_VAR):
                metric.score = score
                test_run.add_llm_test_case(
                    test_case=test_case,
                    metrics=[metric],
                    run_duration=run_duration,
                )

            if not success:
                failed_metrics.append((metric.__name__, score))

    if raise_error and failed_metrics:
        raise AssertionError(
            f"Metrics {', '.join([f'{name} (Score: {score})' for name, score in failed_metrics])} failed."
        )

    return test_results


def assert_test(
    test_cases: Union[LLMTestCase, List[LLMTestCase]],
    metrics: List[BaseMetric],
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


def is_test_passing(
    test_cases: Union[LLMTestCase, List[LLMTestCase]],
    metrics: List[BaseMetric],
    max_retries: int = 1,
    delay: int = 1,
    min_success: int = 1,
) -> bool:
    """Check if a test is passing"""
    test_results = run_test(
        test_cases=test_cases,
        metrics=metrics,
        max_retries=max_retries,
        delay=delay,
        min_success=min_success,
        raise_error=False,
    )
    return all(result.success for result in test_results)
