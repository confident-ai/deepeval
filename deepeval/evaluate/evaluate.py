from typing import Callable, List, Optional, Union, Dict, Any, Awaitable
import time
from rich.console import Console


from deepeval.evaluate.utils import (
    validate_evaluate_inputs,
    print_test_result,
    aggregate_metric_pass_rates,
)
from deepeval.dataset import Golden
from deepeval.prompt import Prompt
from deepeval.test_case.utils import check_valid_test_cases_type
from deepeval.test_run.hyperparameters import process_hyperparameters
from deepeval.test_run.test_run import TestRunResultDisplay
from deepeval.utils import (
    get_or_create_event_loop,
    should_ignore_errors,
    should_skip_on_missing_params,
    should_use_cache,
    should_verbose_print,
)
from deepeval.telemetry import capture_evaluation_run
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
)
from deepeval.metrics.indicator import (
    format_metric_description,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
)
from deepeval.test_run import (
    global_test_run_manager,
    MetricData,
)
from deepeval.utils import get_is_running_deepeval
from deepeval.evaluate.types import EvaluationResult
from deepeval.evaluate.execute import (
    a_execute_agentic_test_cases,
    a_execute_test_cases,
    execute_agentic_test_cases,
    execute_test_cases,
)


def assert_test(
    test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    metrics: List[
        Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric]
    ],
    run_async: bool = True,
):
    if run_async:
        loop = get_or_create_event_loop()
        test_result = loop.run_until_complete(
            a_execute_test_cases(
                [test_case],
                metrics,
                skip_on_missing_params=should_skip_on_missing_params(),
                ignore_errors=should_ignore_errors(),
                use_cache=should_use_cache(),
                verbose_mode=should_verbose_print(),
                throttle_value=0,
                # this doesn't matter for pytest
                max_concurrent=100,
                save_to_disk=get_is_running_deepeval(),
                show_indicator=True,
                _use_bar_indicator=True,
            )
        )[0]
    else:
        test_result = execute_test_cases(
            [test_case],
            metrics,
            skip_on_missing_params=should_skip_on_missing_params(),
            ignore_errors=should_ignore_errors(),
            use_cache=should_use_cache(),
            verbose_mode=should_verbose_print(),
            save_to_disk=get_is_running_deepeval(),
            show_indicator=True,
            _use_bar_indicator=False,
        )[0]

    if not test_result.success:
        failed_metrics_data: List[MetricData] = []
        # even for conversations, test_result right now is just the
        # result for the last message
        for metric_data in test_result.metrics_data:
            if metric_data.error is not None:
                failed_metrics_data.append(metric_data)
            else:
                # This try block is for user defined custom metrics,
                # which might not handle the score == undefined case elegantly
                try:
                    if not metric_data.success:
                        failed_metrics_data.append(metric_data)
                except:
                    failed_metrics_data.append(metric_data)

        failed_metrics_str = ", ".join(
            [
                f"{metrics_data.name} (score: {metrics_data.score}, threshold: {metrics_data.threshold}, strict: {metrics_data.strict_mode}, error: {metrics_data.error})"
                for metrics_data in failed_metrics_data
            ]
        )
        raise AssertionError(f"Metrics: {failed_metrics_str} failed.")


def evaluate(
    goldens: Optional[List[Golden]] = None,
    traceable_callback: Optional[
        Union[Callable[[str], Any], Callable[[str], Awaitable[Any]]]
    ] = None,
    test_cases: Optional[
        Union[
            List[Union[LLMTestCase, MLLMTestCase]], List[ConversationalTestCase]
        ]
    ] = None,
    metrics: Optional[List[BaseMetric]] = None,
    hyperparameters: Optional[Dict[str, Union[str, int, float, Prompt]]] = None,
    # Async config
    run_async: bool = True,
    throttle_value: int = 0,
    max_concurrent: int = 100,
    # Display config
    show_indicator: bool = True,
    print_results: bool = True,
    verbose_mode: Optional[bool] = None,
    display: Optional[TestRunResultDisplay] = TestRunResultDisplay.ALL,
    # Cache config
    write_cache: bool = True,
    use_cache: bool = False,
    # Error config
    ignore_errors: bool = False,
    skip_on_missing_params: bool = False,
    identifier: Optional[str] = None,
) -> EvaluationResult:
    validate_evaluate_inputs(
        goldens=goldens,
        traceable_callback=traceable_callback,
        test_cases=test_cases,
        metrics=metrics,
    )
    if goldens and traceable_callback:
        global_test_run_manager.reset()
        start_time = time.perf_counter()
        with capture_evaluation_run("traceable evaluate()"):
            if run_async:
                loop = get_or_create_event_loop()
                test_results = loop.run_until_complete(
                    a_execute_agentic_test_cases(
                        goldens=goldens,
                        traceable_callback=traceable_callback,
                        ignore_errors=ignore_errors,
                        verbose_mode=verbose_mode,
                        show_indicator=show_indicator,
                        skip_on_missing_params=skip_on_missing_params,
                        throttle_value=throttle_value,
                        identifier=identifier,
                        max_concurrent=max_concurrent,
                    )
                )
            else:
                test_results = execute_agentic_test_cases(
                    goldens=goldens,
                    traceable_callback=traceable_callback,
                    ignore_errors=ignore_errors,
                    verbose_mode=verbose_mode,
                    show_indicator=show_indicator,
                    skip_on_missing_params=skip_on_missing_params,
                    identifier=identifier,
                )
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        global_test_run_manager.wrap_up_test_run(
            run_duration, display_table=True
        )

    elif test_cases and metrics:
        check_valid_test_cases_type(test_cases)

        global_test_run_manager.reset()
        start_time = time.perf_counter()

        if show_indicator:
            console = Console()
            for metric in metrics:
                console.print(
                    format_metric_description(metric, async_mode=run_async)
                )

        with capture_evaluation_run("evaluate()"):
            if run_async:
                loop = get_or_create_event_loop()
                test_results = loop.run_until_complete(
                    a_execute_test_cases(
                        test_cases,
                        metrics,
                        ignore_errors=ignore_errors,
                        use_cache=use_cache,
                        verbose_mode=verbose_mode,
                        save_to_disk=write_cache,
                        show_indicator=show_indicator,
                        skip_on_missing_params=skip_on_missing_params,
                        throttle_value=throttle_value,
                        identifier=identifier,
                        max_concurrent=max_concurrent,
                    )
                )
            else:
                test_results = execute_test_cases(
                    test_cases,
                    metrics,
                    ignore_errors=ignore_errors,
                    use_cache=use_cache,
                    verbose_mode=verbose_mode,
                    save_to_disk=write_cache,
                    skip_on_missing_params=skip_on_missing_params,
                    identifier=identifier,
                    show_indicator=show_indicator,
                )

        end_time = time.perf_counter()
        run_duration = end_time - start_time
        if print_results:
            for test_result in test_results:
                print_test_result(test_result, display)

            aggregate_metric_pass_rates(test_results)

        test_run = global_test_run_manager.get_test_run()
        test_run.hyperparameters = hyperparameters
        global_test_run_manager.save_test_run()
        confident_link = global_test_run_manager.wrap_up_test_run(
            run_duration, display_table=False
        )
        return EvaluationResult(
            test_results=test_results, confident_link=confident_link
        )
