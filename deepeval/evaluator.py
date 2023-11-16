"""Function for running test
"""
import os
import warnings
from typing import List, Optional, Union, Tuple
import time
from dataclasses import dataclass

from deepeval.progress_context import progress_context
from .retry import retry
import copy
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, TestCase
from deepeval.test_run import test_run_manager
import sys


@dataclass
class TestResult:
    """Returned from run_test"""

    success: bool
    metrics: List[BaseMetric]
    input: str
    actual_output: str
    expected_output: str
    context: List[str]

    # def __post_init__(self):
    #     """Ensures score is between 0 and 1 after initialization"""
    #     original_score = self.score
    #     self.score = min(max(0, self.score), 1)
    #     if self.score != original_score:
    #         warnings.warn(
    #             "The score was adjusted to be within the range [0, 1]."
    #         )

    # def __gt__(self, other: "TestResult") -> bool:
    #     """Greater than comparison based on score"""
    #     return self.score > other.score

    # def __lt__(self, other: "TestResult") -> bool:
    #     """Less than comparison based on score"""
    #     return self.score < other.score


def create_test_result(
    test_case: LLMTestCase,
    success: bool,
    metrics: list[BaseMetric],
) -> TestResult:
    if isinstance(test_case, LLMTestCase):
        return TestResult(
            success=success,
            metrics=metrics,
            input=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            context=test_case.context,
        )
    else:
        raise ValueError("TestCase not supported yet.")


def execute_test(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    using_pytest: bool = False,
) -> List[TestResult]:
    test_results: TestResult = []
    test_run_manager.save_to_disk = using_pytest

    for test_case in test_cases:
        success = True
        for metric in metrics:
            test_start_time = time.perf_counter()
            metric.measure(test_case)
            test_end_time = time.perf_counter()
            run_duration = test_end_time - test_start_time

            test_run_manager.get_test_run().add_llm_test_case(
                test_case=test_case,
                metrics=[metric],
                run_duration=run_duration,
            )
            test_run_manager.save_test_run()

            if not metric.is_successful():
                success = False

        test_result = create_test_result(
            test_case, success, copy.deepcopy(metrics)
        )
        test_results.append(test_result)

    return test_results


def run_test(
    test_cases: Union[TestCase, LLMTestCase, List[LLMTestCase]],
    metrics: List[BaseMetric],
) -> List[TestResult]:
    with progress_context("Executing run_test()..."):
        test_results = execute_test(test_cases, metrics, False)
        for test_result in test_results:
            print("\n" + "-" * 50 + "\n")
            print("Metrics")
            for metric in test_result.metrics:
                if not metric.is_successful():
                    print(
                        f"  - ❌ {metric.__name__} (score: {metric.score}, minimum_score: {metric.minimum_score})"
                    )
                else:
                    print(
                        f"  - ✅ {metric.__name__} (score: {metric.score}, minimum_score: {metric.minimum_score})"
                    )

            print("For test case:")
            print(f"  - input: {test_result.input}")
            print(f"  - actual output: {test_result.actual_output}")
            print(f"  - expected output: {test_result.expected_output}")
            print(f"  - context: {test_result.context}")
        print("\n" + "-" * 50)
        return test_results


def assert_test(test_case: LLMTestCase, metrics: List[BaseMetric]):
    # len(execute_test(...)) is always 1 for assert_test
    test_result = execute_test([test_case], metrics, True)[0]
    if not test_result.success:
        failed_metrics = [
            metric
            for metric in test_result.metrics
            if not metric.is_successful()
        ]
        failed_metrics_str = ", ".join(
            [
                f"{metric.__name__} (score: {metric.score}, minimum_score: {metric.minimum_score})"
                for metric in failed_metrics
            ]
        )
        raise AssertionError(f"Metrics {failed_metrics_str} failed.")
