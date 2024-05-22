import os
from typing import List, Optional, Union, Dict
import time
from dataclasses import dataclass

from deepeval.test_run.hyperparameters import process_hyperparameters
from deepeval.utils import (
    drop_and_copy,
    get_or_create_event_loop,
    should_ignore_errors,
    should_use_cache,
)
from deepeval.telemetry import capture_evaluation_run
from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import (
    measure_metrics_with_indicator,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import (
    test_run_manager,
    LLMApiTestCase,
    ConversationalApiTestCase,
    MetricMetadata,
)
from deepeval.utils import get_is_running_deepeval, set_indicator
from deepeval.test_run.cache import (
    test_run_cache_manager,
    Cache,
    CachedTestCase,
    CachedMetricData,
)
from deepeval.tracing import get_trace_stack


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
    if metric.error is not None:
        return MetricMetadata(
            metric=metric.__name__,
            threshold=metric.threshold,
            score=None,
            reason=None,
            success=False,
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=metric.error,
            evaluationCost=metric.evaluation_cost,
        )
    else:
        return MetricMetadata(
            metric=metric.__name__,
            score=metric.score,
            threshold=metric.threshold,
            reason=metric.reason,
            success=metric.is_successful(),
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=None,
            evaluationCost=metric.evaluation_cost,
        )


def create_test_result(
    test_case: Union[LLMApiTestCase, ConversationalApiTestCase],
    metrics: List[BaseMetric],
) -> TestResult:
    if isinstance(test_case, ConversationalApiTestCase):
        tc = test_case.messages[len(test_case.messages) - 1]
    else:
        tc = test_case

    return TestResult(
        success=tc.success,
        metrics=metrics,
        input=tc.input,
        actual_output=tc.actual_output,
        expected_output=tc.expected_output,
        context=tc.context,
        retrieval_context=tc.retrieval_context,
    )


def create_api_test_case(
    test_case: Union[LLMTestCase, ConversationalTestCase],
    index: Optional[int] = None,
    is_message: Optional[bool] = False,
    additional_metadata: Optional[Dict] = None,
    comments: Optional[str] = None,
) -> Union[LLMApiTestCase, ConversationalApiTestCase]:
    if isinstance(test_case, LLMTestCase):
        if is_message:
            success = None
            name = f"message_{index}"
            order = index

            # Manually set the metadata and comments on conversational test case
            # to each individual message (test case)
            test_case.additional_metadata = additional_metadata
            test_case.comments = comments
            traceStack = None
        else:
            success = True
            name = os.getenv(PYTEST_RUN_TEST_NAME, f"test_case_{index}")
            order = test_case._dataset_rank
            traceStack = get_trace_stack()

        return LLMApiTestCase(
            name=name,
            input=test_case.input,
            actualOutput=test_case.actual_output,
            expectedOutput=test_case.expected_output,
            context=test_case.context,
            retrievalContext=test_case.retrieval_context,
            success=success,
            metricsMetadata=None,
            runDuration=None,
            evaluationCost=None,
            order=order,
            additionalMetadata=test_case.additional_metadata,
            comments=test_case.comments,
            traceStack=traceStack,
        )

    elif isinstance(test_case, ConversationalTestCase):
        return ConversationalApiTestCase(
            name=os.getenv(
                PYTEST_RUN_TEST_NAME, f"conversational_test_case_{index}"
            ),
            success=True,
            metricsMetadata=None,
            runDuration=0,
            evaluationCost=None,
            order=test_case._dataset_rank,
            testCases=[
                create_api_test_case(
                    tc,
                    i,
                    True,
                    test_case.additional_metadata,
                    test_case.comments,
                )
                for i, tc in enumerate(test_case.messages)
            ],
        )


def execute_test_cases(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
    metrics: List[BaseMetric],
    ignore_errors: bool,
    use_cache: bool,
    save_to_disk: bool = False,
) -> List[TestResult]:
    test_results: List[TestResult] = []
    test_run_cache_manager.disable_write_cache = save_to_disk == False
    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run()
    for index, test_case in enumerate(test_cases):
        with capture_evaluation_run("test case"):
            if isinstance(test_case, ConversationalTestCase):
                last_message_index = len(test_case.messages) - 1
            else:
                last_message_index = -1

            cached_test_case = None
            if use_cache and isinstance(
                test_case, LLMTestCase
            ):  # for now, don't use cache when it is a conversation
                cached_test_case = test_run_cache_manager.get_cached_test_case(
                    test_case, test_run.hyperparameters
                )

            ##### Metric Calculation #####
            # this can be a converational api or llm api test case
            api_test_case = create_api_test_case(test_case, index)
            new_cached_test_case: CachedTestCase = CachedTestCase()
            test_start_time = time.perf_counter()

            for metric in metrics:
                metric_metadata = None
                # cached_tet_case will always be false for conversationals
                if cached_test_case is not None:
                    cached_metric_data = Cache.get_metric_data(
                        metric, cached_test_case
                    )
                    if cached_metric_data:
                        metric_metadata = cached_metric_data.metric_metadata

                # metric_metadata will always be false for conversationals
                if metric_metadata is None:
                    metric.async_mode = False  # Override metric async
                    try:
                        # if conversational, manually extract the last one for now
                        # In the future, might add support for evaluating any message in a convo
                        if isinstance(test_case, ConversationalTestCase):
                            tc = test_case.messages[last_message_index]
                        else:
                            tc = test_case
                        metric.measure(tc)
                    except Exception as e:
                        if ignore_errors:
                            metric.error = str(e)  # Override metric error
                            metric.success = False  # Override metric success
                        else:
                            raise
                    metric_metadata = create_metric_metadata(metric)

                if isinstance(test_case, ConversationalTestCase):
                    # the index can be dynamic in the future, just not now
                    api_test_case.update(metric_metadata, last_message_index)
                else:
                    api_test_case.update(metric_metadata)

                if metric.error is None and isinstance(
                    test_case, LLMTestCase
                ):  # Only save to cache if metric didn't error and not conversational
                    cache_metric_metadata = create_metric_metadata(metric)
                    cache_metric_metadata.evaluation_cost = (
                        0  # Create copy and save 0 for cost
                    )
                    updated_cached_metric_data = CachedMetricData(
                        metric_metadata=cache_metric_metadata,
                        metric_configuration=Cache.create_metric_configuration(
                            metric
                        ),
                    )
                    new_cached_test_case.cached_metrics_data.append(
                        updated_cached_metric_data
                    )

            test_end_time = time.perf_counter()
            run_duration = test_end_time - test_start_time
            api_test_case.run_duration = run_duration

            ### Update Test Run ###
            test_run_manager.update_test_run(api_test_case, test_case)

            ### Cache Test Run ###
            if isinstance(
                test_case, LLMTestCase
            ):  # only cache if not conversational
                test_run_cache_manager.cache_test_case(
                    test_case,
                    new_cached_test_case,
                    test_run.hyperparameters,
                )
                test_run_cache_manager.cache_test_case(
                    test_case,
                    new_cached_test_case,
                    test_run.hyperparameters,
                    to_temp=True,
                )

            test_result = create_test_result(
                api_test_case, drop_and_copy(metrics, ["model", "embeddings"])
            )
            test_results.append(test_result)

    return test_results


async def a_execute_test_cases(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
    metrics: List[BaseMetric],
    ignore_errors: bool,
    use_cache: bool,
    save_to_disk: bool = False,
) -> List[TestResult]:
    test_results: List[TestResult] = []
    test_run_cache_manager.disable_write_cache = save_to_disk == False
    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run()
    for index, test_case in enumerate(test_cases):
        with capture_evaluation_run("test case"):
            cached_test_case = None
            # only use cache when NOT conversational test case
            if use_cache and isinstance(test_case, LLMTestCase):
                cached_test_case = test_run_cache_manager.get_cached_test_case(
                    test_case,
                    test_run.hyperparameters,
                )

            ##### Metric Calculation #####
            # api test case can be conversational
            api_test_case = create_api_test_case(test_case, index)

            new_cached_test_case: CachedTestCase = CachedTestCase()
            test_start_time = time.perf_counter()
            await measure_metrics_with_indicator(
                metrics, test_case, cached_test_case, ignore_errors
            )
            for metric in metrics:
                metric_metadata = create_metric_metadata(metric)

                if isinstance(test_case, ConversationalTestCase):
                    # index hardcoded as the last message for now
                    api_test_case.update(
                        metric_metadata, len(test_case.messages) - 1
                    )
                else:
                    api_test_case.update(metric_metadata)

                if metric.error is None and isinstance(
                    test_case, LLMTestCase
                ):  # Only save to cache if metric didn't error AND is not conversational
                    cache_metric_metadata = create_metric_metadata(metric)
                    cache_metric_metadata.evaluation_cost = (
                        0  # Create new copy and save 0 for cost
                    )
                    updated_cached_metric_data = CachedMetricData(
                        metric_metadata=cache_metric_metadata,
                        metric_configuration=Cache.create_metric_configuration(
                            metric
                        ),
                    )
                    new_cached_test_case.cached_metrics_data.append(
                        updated_cached_metric_data
                    )

            test_end_time = time.perf_counter()
            run_duration = test_end_time - test_start_time
            api_test_case.run_duration = run_duration

            ### Update Test Run ###
            test_run_manager.update_test_run(api_test_case, test_case)

            ### Cache Test Run ###
            if isinstance(
                test_case, LLMTestCase
            ):  # only cache if not conversational
                test_run_cache_manager.cache_test_case(
                    test_case,
                    new_cached_test_case,
                    test_run.hyperparameters,
                )
                test_run_cache_manager.cache_test_case(
                    test_case,
                    new_cached_test_case,
                    test_run.hyperparameters,
                    to_temp=True,
                )

            test_result = create_test_result(
                api_test_case, drop_and_copy(metrics, ["model", "embeddings"])
            )
            test_results.append(test_result)

    return test_results


def assert_test(
    test_case: Union[LLMTestCase, ConversationalTestCase],
    metrics: List[BaseMetric],
    run_async: bool = True,
):
    # TODO: keep this for now, blocking conversational metrics like KR
    for metric in metrics:
        if not isinstance(metric, BaseMetric):
            raise TypeError("Provided 'metric' must be of type 'BaseMetric'.")

    if run_async:
        loop = get_or_create_event_loop()
        test_result = loop.run_until_complete(
            a_execute_test_cases(
                [test_case],
                metrics,
                ignore_errors=should_ignore_errors(),
                use_cache=should_use_cache(),
                save_to_disk=get_is_running_deepeval(),
            )
        )[0]
    else:
        test_result = execute_test_cases(
            [test_case],
            metrics,
            ignore_errors=should_ignore_errors(),
            use_cache=should_use_cache(),
            save_to_disk=get_is_running_deepeval(),
        )[0]

    if not test_result.success:
        failed_metrics: List[BaseMetric] = []
        # even for conversations, test_result right now is just the
        # result for the last message
        for metric in test_result.metrics:
            if metric.error is not None:
                failed_metrics.append(metric)
            else:
                # This try block is for user defined custom metrics,
                # which might not handle the score == undefined case elegantly
                try:
                    if not metric.is_successful():
                        failed_metrics.append(metric)
                except:
                    failed_metrics.append(metric)

        failed_metrics_str = ", ".join(
            [
                f"{metric.__name__} (score: {metric.score}, threshold: {metric.threshold}, strict: {metric.strict_mode}, error: {metric.error})"
                for metric in failed_metrics
            ]
        )
        raise AssertionError(f"Metrics: {failed_metrics_str} failed.")


def evaluate(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
    metrics: List[BaseMetric],
    hyperparameters: Optional[Dict[str, Union[str, int, float]]] = None,
    run_async: bool = True,
    show_indicator: bool = True,
    print_results: bool = True,
    write_cache: bool = True,
    use_cache: bool = False,
    ignore_errors: bool = False,
):
    if hyperparameters is not None:
        if (
            hyperparameters.get("model") is None
            or hyperparameters.get("prompt template") is None
        ):
            raise ValueError(
                "A `model` and `prompt template` key must be provided when logging `hyperparameters`."
            )
        hyperparameters = process_hyperparameters(hyperparameters)

    set_indicator(show_indicator)

    # TODO: keep this for now, blocking conversational metrics like KR
    for metric in metrics:
        if not isinstance(metric, BaseMetric):
            raise TypeError("Provided 'metric' must be of type 'BaseMetric'.")

    test_run_manager.reset()
    start_time = time.perf_counter()
    if print_results:
        print("Evaluating test cases...")

    with capture_evaluation_run("evaluate()"):
        if run_async:
            loop = get_or_create_event_loop()
            test_results = loop.run_until_complete(
                a_execute_test_cases(
                    test_cases,
                    metrics,
                    ignore_errors=ignore_errors,
                    use_cache=use_cache,
                    save_to_disk=write_cache,
                )
            )
        else:
            test_results = execute_test_cases(
                test_cases,
                metrics,
                ignore_errors=ignore_errors,
                use_cache=use_cache,
                save_to_disk=write_cache,
            )

    end_time = time.perf_counter()
    run_duration = end_time - start_time
    if print_results:
        for test_result in test_results:
            print_test_result(test_result)

        aggregate_metric_pass_rates(test_results)

    test_run = test_run_manager.get_test_run()
    test_run.hyperparameters = hyperparameters
    test_run_manager.save_test_run()
    test_run_manager.wrap_up_test_run(run_duration, display_table=False)
    return test_results


def print_test_result(test_result: TestResult):
    print("")
    print("=" * 70 + "\n")
    print("Metrics Summary\n")
    for metric in test_result.metrics:
        successful = True
        if metric.error is not None:
            successful = False
        else:
            # This try block is for user defined custom metrics,
            # which might not handle the score == undefined case elegantly
            try:
                if not metric.is_successful():
                    successful = False
            except:
                successful = False

        if not successful:
            print(
                f"  - ❌ {metric.__name__} (score: {metric.score}, threshold: {metric.threshold}, strict: {metric.strict_mode}, evaluation model: {metric.evaluation_model}, reason: {metric.reason}, error: {metric.error})"
            )
        else:
            print(
                f"  - ✅ {metric.__name__} (score: {metric.score}, threshold: {metric.threshold}, strict: {metric.strict_mode}, evaluation model: {metric.evaluation_model}, reason: {metric.reason}, error: {metric.error})"
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


def aggregate_metric_pass_rates(test_results: List[TestResult]) -> dict:
    metric_counts = {}
    metric_successes = {}

    for result in test_results:
        for metric in result.metrics:
            metric_name = metric.__class__.__name__
            if metric_name not in metric_counts:
                metric_counts[metric_name] = 0
                metric_successes[metric_name] = 0
            metric_counts[metric_name] += 1
            if metric.success:
                metric_successes[metric_name] += 1

    metric_pass_rates = {
        metric: (metric_successes[metric] / metric_counts[metric])
        for metric in metric_counts
    }

    print("\n" + "=" * 70 + "\n")
    print("Overall Metric Pass Rates\n")
    for metric, pass_rate in metric_pass_rates.items():
        print(f"{metric}: {pass_rate:.2%} pass rate")
    print("\n" + "=" * 70 + "\n")

    return metric_pass_rates
