"""Function for running test
"""
import os
import warnings
from typing import List, Optional, Union
import time
from dataclasses import dataclass
from .retry import retry

from .metrics import BaseMetric
from .test_case import LLMTestCase, TestCase
from deepeval.test_run import test_run_manager
import sys


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
    metric: float,
) -> TestResult:
    if isinstance(test_case, LLMTestCase):
        return TestResult(
            success=success,
            score=metric.score,
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
    for test_case in test_cases:
        failed_metrics = []
        for metric in metrics:
            test_start_time = time.perf_counter()
            # score = metric.measure(test_case)
            metric.score = metric.measure(test_case)
            success = metric.is_successful()
            test_end_time = time.perf_counter()
            run_duration = test_end_time - test_start_time
            # metric.score = score

            test_run_manager.get_test_run().add_llm_test_case(
                test_case=test_case,
                metrics=[metric],
                run_duration=run_duration,
            )
            test_run_manager.save_test_run()

            test_result = create_test_result(test_case, success, metric)
            test_results.append(test_result)
            if not success:
                failed_metrics.append((metric.__name__, metric.score))

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
