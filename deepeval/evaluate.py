import os
from typing import List, Optional
import time
from dataclasses import dataclass
import json

from deepeval.utils import drop_and_copy, get_or_create_event_loop
from deepeval.telemetry import capture_evaluation_count
from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import (
    measure_metrics_with_indicator,
)
from deepeval.test_case import LLMTestCase
from deepeval.tracing import get_trace_stack
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import test_run_manager, APITestCase, MetricsMetadata
from deepeval.test_run.cache import test_run_cache_manager
from deepeval.utils import get_is_running_deepeval, set_indicator


@dataclass
class TestResult:
    """Returned from run_test"""

    success: bool
    metrics: List[BaseMetric]
    input: str
    actual_output: str
    expected_output: str
    context: List[str]
    retrieval_context: List[str]


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
            retrieval_context=test_case.retrieval_context,
        )
    else:
        raise ValueError("TestCase not supported yet.")


def create_api_test_case(
    test_case: LLMTestCase,
    index: Optional[int] = None,
) -> APITestCase:
    return APITestCase(
        name=os.getenv(PYTEST_RUN_TEST_NAME, f"test_case_{index}"),
        input=test_case.input,
        actualOutput=test_case.actual_output,
        expectedOutput=test_case.expected_output,
        success=True,
        metricsMetadata=[],
        runDuration=0,
        latency=test_case.latency,
        cost=test_case.cost,
        context=test_case.context,
        retrievalContext=test_case.retrieval_context,
        traceStack=get_trace_stack(),
        id=test_case.id,
    )


def generate_cache_key(test_case: LLMTestCase):
        # Explicitly handle None values for context and retrieval_context
        context = sorted(test_case.context) if test_case.context else []
        retrieval_context = sorted(test_case.retrieval_context) if test_case.retrieval_context else []

        # Create a dictionary with the relevant fields
        cache_data = {
            "input": test_case.input,
            "expected_output": test_case.expected_output,
            "context": context,
            "retrieval_context": retrieval_context,
        }
        
        # Sort the dictionary by key to ensure consistent ordering
        sorted_cache_data = dict(sorted(cache_data.items()))
        
        # Convert the sorted dictionary to a JSON string to use as a cache key
        # JSON serialization ensures that the data structure is converted to a string format that can be hashed
        cache_key = json.dumps(sorted_cache_data)
        
        return cache_key 

def execute_test_cases(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    save_to_disk: bool = False,
) -> List[TestResult]:
    
    test_results: List[TestResult] = []
    test_run_manager.save_to_disk = save_to_disk

    for index, test_case in enumerate(test_cases):

        test_run = test_run_manager.get_test_run()
        cached_test_run = test_run_cache_manager.get_cached_test_run()
        key = generate_cache_key(test_case=test_case)
        look_up_map = cached_test_run.test_cases_lookup_map

        #########################################
        # Get cached data if available
        ##########################################
        already_in_cache = (key in look_up_map)
        hyperparameter_same = cached_test_run.hyperparameters == test_run.hyperparameters
        actual_output_exists = False
        metrics_metadata_exists = False
        cached_api_test_case = None

        if already_in_cache:
            cached_api_test_case = look_up_map[key]
            actual_output_exists = cached_api_test_case.actual_output != None
            if actual_output_exists:
                metrics_metadata_exists = cached_api_test_case.metrics_metadata != None
        can_use_cache = (already_in_cache and hyperparameter_same and actual_output_exists and metrics_metadata_exists)

        cached_metrics_metadata = {} 
        if can_use_cache:
            metrics_metadata = look_up_map[key].metrics_metadata
            cached_metrics_metadata = {metadata.metric: metadata for metadata in metrics_metadata}

        #########################################
        # Calculate remaining metrics (if not cached)
        #########################################
        success = True
        api_test_case: APITestCase = create_api_test_case(test_case, index)
        test_start_time = time.perf_counter()


        for metric in metrics:
            metric_metadata = None
            if metric.__name__ in cached_metrics_metadata:
                metric_metadata = cached_metrics_metadata[metric.__name__]
            else:
                metric.async_mode = False # Override metric async
                metric.measure(test_case)
                metric_metadata = MetricsMetadata(
                    metric=metric.__name__,
                    score=metric.score,
                    threshold=metric.threshold,
                    reason=metric.reason,
                    success=metric.is_successful(),
                    evaluationModel=metric.evaluation_model,
                )
                if cached_api_test_case:
                    cached_api_test_case.metrics_metadata.append(metric_metadata)

            api_test_case.metrics_metadata.append(metric_metadata)
            if metric_metadata.success is False:
                success = False

        test_end_time = time.perf_counter()
        run_duration = test_end_time - test_start_time
        api_test_case.run_duration = run_duration
        api_test_case.success = success

        #########################################
        # Update test_run_manager
        #########################################
        test_run.test_cases.append(api_test_case)
        test_run.dataset_alias = test_case.dataset_alias
        test_run_manager.save_test_run()
        test_result = create_test_result(
            test_case, success, drop_and_copy(metrics, ["model", "embeddings"])
        )
        test_results.append(test_result)

        #########################################
        # Update test_run_cache_manager
        #########################################
        cached_api_test_case = api_test_case if cached_api_test_case is None else cached_api_test_case
        cached_test_run.test_cases_lookup_map[key] = cached_api_test_case
        cached_test_run.hyperparameters = test_run.hyperparameters
        test_run_cache_manager.save_cached_test_run()


    return test_results


async def a_execute_test_cases(
        
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    save_to_disk: bool = False,
) -> List[TestResult]:
    
    test_results: List[TestResult] = []
    test_run_manager.save_to_disk = save_to_disk

    for index, test_case in enumerate(test_cases):

        test_run = test_run_manager.get_test_run()
        cached_test_run = test_run_cache_manager.get_cached_test_run()
        key = generate_cache_key(test_case=test_case)
        look_up_map = cached_test_run.test_cases_lookup_map

        #########################################
        # Get cached data if available
        ##########################################
        already_in_cache = (key in look_up_map)
        hyperparameter_same = cached_test_run.hyperparameters == test_run.hyperparameters
        actual_output_exists = False
        metrics_metadata_exists = False
        cached_api_test_case = None

        if already_in_cache:
            cached_api_test_case = look_up_map[key]
            actual_output_exists = cached_api_test_case.actual_output != None
            if actual_output_exists:
                metrics_metadata_exists = cached_api_test_case.metrics_metadata != None
        can_use_cache = (already_in_cache and hyperparameter_same and actual_output_exists and metrics_metadata_exists)

        cached_metrics_metadata = {} 
        if can_use_cache:
            metrics_metadata = look_up_map[key].metrics_metadata
            cached_metrics_metadata = {metadata.metric: metadata for metadata in metrics_metadata}

        #########################################
        # Calculate remaining metrics (if not cached)
        #########################################
        success = True
        api_test_case: APITestCase = create_api_test_case(test_case, index)
        test_start_time = time.perf_counter()

        await measure_metrics_with_indicator(metrics, test_case, list(cached_metrics_metadata.keys()))
        for metric in metrics:
            metric_metadata = None
            if metric.__name__ in cached_metrics_metadata:
                metric_metadata = cached_metrics_metadata[metric.__name__]
            else:
                metric_metadata = MetricsMetadata(
                    metric=metric.__name__,
                    score=metric.score,
                    threshold=metric.threshold,
                    reason=metric.reason,
                    success=metric.is_successful(),
                    evaluationModel=metric.evaluation_model,
                )
                if cached_api_test_case:
                    cached_api_test_case.metrics_metadata.append(metric_metadata)

            api_test_case.metrics_metadata.append(metric_metadata)
            if metric_metadata.success is False:
                success = False

        test_end_time = time.perf_counter()
        run_duration = test_end_time - test_start_time
        api_test_case.run_duration = run_duration
        api_test_case.success = success

        #########################################
        # Update test_run_manager
        #########################################
        test_run.test_cases.append(api_test_case)
        test_run.dataset_alias = test_case.dataset_alias
        test_run_manager.save_test_run()
        test_result = create_test_result(
            test_case, success, drop_and_copy(metrics, ["model", "embeddings"])
        )
        test_results.append(test_result)

        #########################################
        # Update test_run_cache_manager
        #########################################
        cached_api_test_case = api_test_case if cached_api_test_case is None else cached_api_test_case
        cached_test_run.test_cases_lookup_map[key] = cached_api_test_case
        cached_test_run.hyperparameters = test_run.hyperparameters
        test_run_cache_manager.save_cached_test_run()

    return test_results


def assert_test(
    test_case: LLMTestCase, metrics: List[BaseMetric], run_async: bool = True
):
    # TODO: refactor
    for metric in metrics:
        if not isinstance(metric, BaseMetric):
            raise TypeError("Provided 'metric' must be of type 'BaseMetric'.")

    # TODO: refactor
    if not isinstance(test_case, LLMTestCase):
        raise TypeError("'test_case' must be an instance of 'LLMTestCase'.")
    if run_async:
        loop = get_or_create_event_loop()
        test_result = loop.run_until_complete(
            a_execute_test_cases(
                [test_case], metrics, get_is_running_deepeval()
            )
        )[0]
    else:
        test_result = execute_test_cases(
            [test_case], metrics, get_is_running_deepeval()
        )[0]
        
    if not test_result.success:
        failed_metrics = [
            metric
            for metric in test_result.metrics
            if not metric.is_successful()
        ]
        failed_metrics_str = ", ".join(
            [
                f"{metric.__name__} (score: {metric.score}, threshold: {metric.threshold}, strict: {metric.strict_mode})"
                for metric in failed_metrics
            ]
        )
        raise AssertionError(f"Metrics {failed_metrics_str} failed.")


def evaluate(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    run_async: bool = True,
    show_indicator: bool = True,
    print_results: bool = True,
):
    set_indicator(show_indicator)

    # TODO: refactor
    for metric in metrics:
        if not isinstance(metric, BaseMetric):
            raise TypeError("Provided 'metric' must be of type 'BaseMetric'.")

    # TODO: refactor
    for test_case in test_cases:
        if not isinstance(test_case, LLMTestCase):
            raise TypeError(
                "Provided `test_cases` must be of type 'List[LLMTestCase]'."
            )

    test_run_manager.reset()

    if print_results:
        print("Evaluating test cases...")
    if run_async:
        loop = get_or_create_event_loop()
        test_results = loop.run_until_complete(
            a_execute_test_cases(test_cases, metrics, True)
        )
    else:
        
        test_results = execute_test_cases(test_cases, metrics, True)

    capture_evaluation_count()

    if print_results:
        for test_result in test_results:
            print_test_result(test_result)

        print("")
        print("-" * 70)
    test_run_manager.wrap_up_test_run(display_table=False)
    return test_results


def print_test_result(test_result: TestResult):
    print("")
    print("=" * 70 + "\n")
    print("Metrics Summary\n")
    for metric in test_result.metrics:
        if not metric.is_successful():
            print(
                f"  - ❌ {metric.__name__} (score: {metric.score}, threshold: {metric.threshold}, strict: {metric.strict_mode}, evaluation model: {metric.evaluation_model}, reason: {metric.reason})"
            )
        else:
            print(
                f"  - ✅ {metric.__name__} (score: {metric.score}, threshold: {metric.threshold}, strict: {metric.strict_mode}, evaluation model: {metric.evaluation_model}, reason: {metric.reason})"
            )
        if metric.score_breakdown:
            for metric_name, score in metric.score_breakdown.items():
                print(f"      - {metric_name} (score: {score})")

    print("")
    print("For test case:\n")
    print(f"  - input: {test_result.input}")
    print(f"  - actual output: {test_result.actual_output}")
    print(f"  - expected output: {test_result.expected_output}")
    print(f"  - context: {test_result.context}")
    print(f"  - retrieval context: {test_result.retrieval_context}")
