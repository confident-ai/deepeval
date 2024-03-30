import os
from typing import List, Optional
import time
from dataclasses import dataclass
import sys

from deepeval.utils import (
    drop_and_copy,
    get_or_create_event_loop,
    should_use_cache,
)
from deepeval.telemetry import capture_evaluation_count
from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import (
    measure_metrics_with_indicator,
)
from deepeval.test_case import LLMTestCase
from deepeval.tracing import get_trace_stack
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import test_run_manager, APITestCase, MetricMetadata
from deepeval.utils import get_is_running_deepeval, set_indicator
from deepeval.test_run.cache import (
    test_run_cache_manager,
    Cache,
    CachedTestCase,
    CachedMetricData,
)


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


def create_metric_metadata(metric: BaseMetric) -> MetricMetadata:
    return MetricMetadata(
        metric=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        reason=metric.reason,
        success=metric.is_successful(),
        strictMode=metric.strict_mode,
        evaluationModel=metric.evaluation_model,
    )


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
    use_cache: bool,
    save_to_disk: bool = False,
) -> List[TestResult]:
    test_results: List[TestResult] = []
    test_run_cache_manager.disable_write_cache = save_to_disk == False
    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run()
    for index, test_case in enumerate(test_cases):
        success = True
        cached_test_case = None
        if use_cache:
            cached_test_case = test_run_cache_manager.get_cached_test_case(
                test_case,
                test_run.hyperparameters,
                test_run.model,
                test_run.user_prompt_template,
            )

        ##### Metric Calculation #####
        api_test_case: APITestCase = create_api_test_case(test_case, index)
        new_cached_test_case: CachedTestCase = CachedTestCase()
        test_start_time = time.perf_counter()

        for metric in metrics:
            metric_metadata = None
            if cached_test_case is not None:
                cached_metric_data = Cache.get_metric_data(
                    metric, cached_test_case
                )
                if cached_metric_data:
                    metric_metadata = cached_metric_data.metric_metadata

            if metric_metadata is None:
                metric.async_mode = False  # Override metric async
                metric.measure(test_case)
                metric_metadata = create_metric_metadata(metric)

            if metric_metadata.success is False:
                success = False

            api_test_case.metrics_metadata.append(metric_metadata)
            updated_cached_metric_data = CachedMetricData(
                metric_metadata=metric_metadata,
                metric_configuration=Cache.create_metric_configuration(metric),
            )
            new_cached_test_case.cached_metrics_data.append(
                updated_cached_metric_data
            )

        test_end_time = time.perf_counter()
        run_duration = test_end_time - test_start_time
        api_test_case.run_duration = run_duration
        api_test_case.success = success

        ### Save Test Run ###
        test_run = test_run_manager.get_test_run()
        test_run.test_cases.append(api_test_case)
        test_run.dataset_alias = test_case.dataset_alias
        test_run_manager.save_test_run()

        ### Cache Test Run ###
        test_run_cache_manager.cache_test_case(
            test_case,
            new_cached_test_case,
            test_run.hyperparameters,
            test_run.model,
            test_run.user_prompt_template,
        )
        test_run_cache_manager.cache_test_case(
            test_case,
            new_cached_test_case,
            test_run.hyperparameters,
            test_run.model,
            test_run.user_prompt_template,
            to_temp=True,
        )

        test_result = create_test_result(
            test_case, success, drop_and_copy(metrics, ["model", "embeddings"])
        )
        test_results.append(test_result)

    return test_results


async def a_execute_test_cases(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    use_cache: bool,
    save_to_disk: bool = False,
) -> List[TestResult]:
    test_results: List[TestResult] = []
    test_run_cache_manager.disable_write_cache = save_to_disk == False
    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run()
    for index, test_case in enumerate(test_cases):
        success = True
        cached_test_case = None
        if use_cache:
            cached_test_case = test_run_cache_manager.get_cached_test_case(
                test_case,
                test_run.hyperparameters,
                test_run.model,
                test_run.user_prompt_template,
            )

        ##### Metric Calculation #####
        api_test_case: APITestCase = create_api_test_case(test_case, index)
        new_cached_test_case: CachedTestCase = CachedTestCase()
        test_start_time = time.perf_counter()
        await measure_metrics_with_indicator(
            metrics,
            test_case,
            cached_test_case,
        )
        for metric in metrics:
            metric_metadata = create_metric_metadata(metric)

            if metric_metadata.success is False:
                success = False

            api_test_case.metrics_metadata.append(metric_metadata)
            updated_cached_metric_data = CachedMetricData(
                metric_metadata=metric_metadata,
                metric_configuration=Cache.create_metric_configuration(metric),
            )
            new_cached_test_case.cached_metrics_data.append(
                updated_cached_metric_data
            )

        test_end_time = time.perf_counter()
        run_duration = test_end_time - test_start_time
        api_test_case.run_duration = run_duration
        api_test_case.success = success

        ### Save Test Run ###
        test_run = test_run_manager.get_test_run()
        test_run.test_cases.append(api_test_case)
        test_run.dataset_alias = test_case.dataset_alias
        test_run_manager.save_test_run()

        ### Cache Test Run ###
        test_run_cache_manager.cache_test_case(
            test_case,
            new_cached_test_case,
            test_run.hyperparameters,
            test_run.model,
            test_run.user_prompt_template,
        )
        test_run_cache_manager.cache_test_case(
            test_case,
            new_cached_test_case,
            test_run.hyperparameters,
            test_run.model,
            test_run.user_prompt_template,
            to_temp=True,
        )

        test_result = create_test_result(
            test_case, success, drop_and_copy(metrics, ["model", "embeddings"])
        )
        test_results.append(test_result)

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
                [test_case],
                metrics,
                use_cache=should_use_cache(),
                save_to_disk=get_is_running_deepeval(),
            )
        )[0]
    else:
        test_result = execute_test_cases(
            [test_case],
            metrics,
            use_cache=should_use_cache(),
            save_to_disk=get_is_running_deepeval(),
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
    use_cache: bool = False,
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
            a_execute_test_cases(
                test_cases, metrics, use_cache=use_cache, save_to_disk=True
            )
        )
    else:
        test_results = execute_test_cases(
            test_cases, metrics, use_cache=use_cache, save_to_disk=True
        )

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
