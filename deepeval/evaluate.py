import os
from typing import List, Optional, Union
import time
from dataclasses import dataclass
import json
from enum import Enum
from langchain_core.embeddings import Embeddings

from deepeval.utils import drop_and_copy, get_or_create_event_loop
from deepeval.telemetry import capture_evaluation_count
from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import (
    measure_metrics_with_indicator,
)
from deepeval.test_case import LLMTestCase
from deepeval.types import Languages
from deepeval.tracing import get_trace_stack
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import test_run_manager, APITestCase, MetricsMetadata
from deepeval.utils import get_is_running_deepeval, set_indicator
from deepeval.test_run.cache import test_run_cache_manager, Cache, CachedAPITestCase, CachedMetricsMetadata

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
    metrics: List[BaseMetric],
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

def execute_test_cases(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    save_to_disk: bool = False,
    use_cache: bool = True
) -> List[TestResult]:
    
    test_results: List[TestResult] = []
    test_run_manager.save_to_disk = save_to_disk

    for index, test_case in enumerate(test_cases):

        ##### Cache Logic #####
        test_run = test_run_manager.get_test_run()
        cached_api_test_case = None
        cached_test_run = None
        temp_cached_test_run = None
        key = Cache.generate_cache_key(test_case=test_case)

        if use_cache:
            cached_test_run = test_run_cache_manager.get_cached_test_run()
            temp_cached_test_run = test_run_cache_manager.get_temp_cached_test_run()
            
            if cached_test_run.hyperparameters == test_run.hyperparameters:
                cached_api_test_case = cached_test_run.get_cached_api_test_case(key)

        ##### Metric Calculation #####
        success = True
        api_test_case: APITestCase = create_api_test_case(test_case, index)
        new_cached_api_test_case: CachedAPITestCase = Cache.create_cached_api_test_case()
        test_start_time = time.perf_counter()

        for metric in metrics:
            metric_config = Cache.create_metric_configuration(metric)
            metric_metadata = Cache.get_metrics_metadata_from_cache(metric, cached_api_test_case)
            if not metric_metadata:
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
            api_test_case.metrics_metadata.append(metric_metadata)

            cached_metrics_metadata = CachedMetricsMetadata(
                metric_metadata=metric_metadata,
                metric_configuration = metric_config
            )
            new_cached_api_test_case.cached_metrics_metadata.append(cached_metrics_metadata)
            if metric_metadata.success is False:
                success = False

        test_end_time = time.perf_counter()
        run_duration = test_end_time - test_start_time
        api_test_case.run_duration = run_duration
        api_test_case.success = success

        test_run.test_cases.append(api_test_case)
        test_run.dataset_alias = test_case.dataset_alias
        test_run_manager.save_test_run()
        test_result = create_test_result(
            test_case, success, drop_and_copy(metrics, ["model", "embeddings"])
        )
        test_results.append(test_result)

        ##### Cache Logic #####
        if use_cache:
            temp_cached_test_run.test_cases_lookup_map[key] = new_cached_api_test_case
            cached_test_run.test_cases_lookup_map[key] = new_cached_api_test_case
            cached_test_run.hyperparameters = test_run.hyperparameters
            temp_cached_test_run.hyperparameters = test_run.hyperparameters
            test_run_cache_manager.save_cached_test_run()
            test_run_cache_manager.save_temp_cached_test_run()

    return test_results


async def a_execute_test_cases(
        
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    save_to_disk: bool = False,
    use_cache: bool = True
) -> List[TestResult]:
    
    test_results: List[TestResult] = []
    test_run_manager.save_to_disk = save_to_disk

    for index, test_case in enumerate(test_cases):

        ##### Cache Logic #####
        test_run = test_run_manager.get_test_run()
        cached_api_test_case = None
        cached_test_run = None
        temp_cached_test_run = None
        key = Cache.generate_cache_key(test_case=test_case)

        if use_cache:
            cached_test_run = test_run_cache_manager.get_cached_test_run()
            temp_cached_test_run = test_run_cache_manager.get_temp_cached_test_run()
            
            if cached_test_run.hyperparameters == test_run.hyperparameters:
                cached_api_test_case = cached_test_run.get_cached_api_test_case(key)

        ##### Metric Calculation #####
        success = True
        api_test_case: APITestCase = create_api_test_case(test_case, index)
        new_cached_api_test_case: CachedAPITestCase = Cache.create_cached_api_test_case()
        test_start_time = time.perf_counter()
        
        metric_configs = [Cache.create_metric_configuration(metric) for metric in metrics]
        await measure_metrics_with_indicator(metrics, test_case, cached_api_test_case, Cache.get_metrics_metadata_from_cache)
        for (i, metric) in enumerate(metrics):
            metric_metadata = Cache.get_metrics_metadata_from_cache(metric, cached_api_test_case)
            if not metric_metadata:
                metric_metadata = MetricsMetadata(
                metric=metric.__name__,
                score=metric.score,
                threshold=metric.threshold,
                reason=metric.reason,
                success=metric.is_successful(),
                evaluationModel=metric.evaluation_model,
            )
            api_test_case.metrics_metadata.append(metric_metadata)

            cached_metrics_metadata = CachedMetricsMetadata(
                metric_metadata=metric_metadata,
                metric_configuration = metric_configs[i]
            )

            new_cached_api_test_case.cached_metrics_metadata.append(cached_metrics_metadata)
            if metric_metadata.success is False:
                success = False

        test_end_time = time.perf_counter()
        run_duration = test_end_time - test_start_time
        api_test_case.run_duration = run_duration
        api_test_case.success = success

        test_run.test_cases.append(api_test_case)
        test_run.dataset_alias = test_case.dataset_alias
        test_run_manager.save_test_run()
        test_result = create_test_result(
            test_case, success, drop_and_copy(metrics, ["model", "embeddings"])
        )
        test_results.append(test_result)

        ##### Cache Logic #####
        if use_cache:
            temp_cached_test_run.test_cases_lookup_map[key] = new_cached_api_test_case
            cached_test_run.test_cases_lookup_map[key] = new_cached_api_test_case
            cached_test_run.hyperparameters = test_run.hyperparameters
            temp_cached_test_run.hyperparameters = test_run.hyperparameters
            test_run_cache_manager.save_cached_test_run()
            test_run_cache_manager.save_temp_cached_test_run()

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
                [test_case], metrics, get_is_running_deepeval(), test_run_manager.use_cache
            )
        )[0]
    else:
        test_result = execute_test_cases(
            [test_case], metrics, get_is_running_deepeval(), test_run_manager.use_cache
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
