"""Function for running test
"""

from typing import List
import time
from dataclasses import dataclass
import copy

from deepeval.progress_context import progress_context
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import test_run_manager


@dataclass
class TestResult:
    """Returned from run_test"""

    success: bool
    metrics: List[BaseMetric]
    input: str
    actual_output: str
    expected_output: str
    context: List[str]


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
    save_to_disk: bool = False,
) -> List[TestResult]:
    test_results: TestResult = []
    test_run_manager.save_to_disk = save_to_disk
    count = 0
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
                index=count,
            )
            test_run_manager.save_test_run()

            if not metric.is_successful():
                success = False

        count += 1
        test_result = create_test_result(
            test_case, success, copy.deepcopy(metrics)
        )
        test_results.append(test_result)

    return test_results


def run_test(
    test_case: LLMTestCase,
    metrics: List[BaseMetric],
) -> List[TestResult]:
    with progress_context("Executing run_test()..."):
        test_result = execute_test([test_case], metrics, False)[0]
        print_test_result(test_result)
        print("\n" + "-" * 70)
        return test_result


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


def evaluate(test_cases: List[LLMTestCase], metrics: List[BaseMetric]):
    with progress_context("Evaluating testcases..."):
        test_results = execute_test(test_cases, metrics, True)
        for test_result in test_results:
            print_test_result(test_result)
        print("\n" + "-" * 70)

        test_run_manager.wrap_up_test_run(display_table=False)
        return test_results


def print_test_result(test_result: TestResult):
    print("\n" + "=" * 70 + "\n")
    print("Metrics Summary\n")
    for metric in test_result.metrics:
        if not metric.is_successful():
            print(
                f"  - ❌ {metric.__name__} (score: {metric.score}, minimum_score: {metric.minimum_score})"
            )
        else:
            print(
                f"  - ✅ {metric.__name__} (score: {metric.score}, minimum_score: {metric.minimum_score})"
            )

    print("\nFor test case:\n")
    print(f"  - input: {test_result.input}")
    print(f"  - actual output: {test_result.actual_output}")
    print(f"  - expected output: {test_result.expected_output}")
    print(f"  - context: {test_result.context}")
