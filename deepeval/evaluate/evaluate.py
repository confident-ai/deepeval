from typing import Callable, List, Optional, Union, Dict, Any, Awaitable
import time
from rich.console import Console

from deepeval.evaluate.configs import (
    AsyncConfig,
    DisplayConfig,
    CacheConfig,
    ErrorConfig,
)
from deepeval.evaluate.utils import (
    validate_assert_test_inputs,
    validate_evaluate_inputs,
    print_test_result,
    aggregate_metric_pass_rates,
    write_test_result_to_file,
)
from deepeval.dataset import Golden
from deepeval.prompt import Prompt
from deepeval.test_case.utils import check_valid_test_cases_type
from deepeval.test_run.hyperparameters import process_hyperparameters
from deepeval.test_run.test_run import TEMP_FILE_PATH
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
    test_case: Optional[
        Union[LLMTestCase, ConversationalTestCase, MLLMTestCase]
    ] = None,
    metrics: Optional[
        Union[
            List[BaseMetric],
            List[BaseConversationalMetric],
            List[BaseMultimodalMetric],
        ]
    ] = None,
    golden: Optional[Golden] = None,
    observed_callback: Optional[
        Union[Callable[[str], Any], Callable[[str], Awaitable[Any]]]
    ] = None,
    run_async: bool = True,
):
    validate_assert_test_inputs(
        golden=golden,
        observed_callback=observed_callback,
        test_case=test_case,
        metrics=metrics,
    )

    if golden and observed_callback:
        if run_async:
            loop = get_or_create_event_loop()
            test_result = loop.run_until_complete(
                a_execute_agentic_test_cases(
                    goldens=[golden],
                    observed_callback=observed_callback,
                    ignore_errors=should_ignore_errors(),
                    verbose_mode=should_verbose_print(),
                    show_indicator=True,
                    save_to_disk=get_is_running_deepeval(),
                    skip_on_missing_params=should_skip_on_missing_params(),
                    throttle_value=0,
                    max_concurrent=100,
                    _use_bar_indicator=True,
                    _is_assert_test=True,
                )
            )[0]
        else:
            test_result = execute_agentic_test_cases(
                goldens=[golden],
                observed_callback=observed_callback,
                ignore_errors=should_ignore_errors(),
                verbose_mode=should_verbose_print(),
                show_indicator=True,
                save_to_disk=get_is_running_deepeval(),
                skip_on_missing_params=should_skip_on_missing_params(),
                _use_bar_indicator=False,
                _is_assert_test=True,
            )[0]

    elif test_case and metrics:
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
                    _is_assert_test=True,
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
                _is_assert_test=True,
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
                f"{metrics_data.name} (score: {metrics_data.score}, threshold: {metrics_data.threshold}, strict: {metrics_data.strict_mode}, error: {metrics_data.error}, reason: {metrics_data.reason})"
                for metrics_data in failed_metrics_data
            ]
        )
        raise AssertionError(f"Metrics: {failed_metrics_str} failed.")


def evaluate(
    # without tracing
    test_cases: Optional[
        Union[
            List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
        ]
    ] = None,
    metrics: Optional[
        Union[
            List[BaseMetric],
            List[BaseConversationalMetric],
            List[BaseMultimodalMetric],
        ]
    ] = None,
    hyperparameters: Optional[Dict[str, Union[str, int, float, Prompt]]] = None,
    # with tracing
    goldens: Optional[List[Golden]] = None,
    observed_callback: Optional[
        Union[Callable[[str], Any], Callable[[str], Awaitable[Any]]]
    ] = None,
    # agnostic
    identifier: Optional[str] = None,
    # Configs
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
) -> EvaluationResult:
    validate_evaluate_inputs(
        goldens=goldens,
        observed_callback=observed_callback,
        test_cases=test_cases,
        metrics=metrics,
    )
    if goldens and observed_callback:
        global_test_run_manager.reset()
        # global_test_run_manager.save_to_disk = True
        start_time = time.perf_counter()
        with capture_evaluation_run("traceable evaluate()"):
            if async_config.run_async:
                loop = get_or_create_event_loop()
                test_results = loop.run_until_complete(
                    a_execute_agentic_test_cases(
                        goldens=goldens,
                        observed_callback=observed_callback,
                        ignore_errors=error_config.ignore_errors,
                        verbose_mode=display_config.verbose_mode,
                        show_indicator=display_config.show_indicator,
                        skip_on_missing_params=error_config.skip_on_missing_params,
                        throttle_value=async_config.throttle_value,
                        identifier=identifier,
                        max_concurrent=async_config.max_concurrent,
                        save_to_disk=cache_config.write_cache,
                    )
                )
            else:
                test_results = execute_agentic_test_cases(
                    goldens=goldens,
                    observed_callback=observed_callback,
                    ignore_errors=error_config.ignore_errors,
                    verbose_mode=display_config.verbose_mode,
                    show_indicator=display_config.show_indicator,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                    identifier=identifier,
                    save_to_disk=cache_config.write_cache,
                )
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        if display_config.print_results:
            for test_result in test_results:
                print_test_result(test_result, display_config.display_option)
                aggregate_metric_pass_rates(test_results)
        if display_config.file_output_dir is not None:
            for test_result in test_results:
                write_test_result_to_file(
                    test_result,
                    display_config.display_option,
                    display_config.file_output_dir,
                )

        confident_link = global_test_run_manager.wrap_up_test_run(
            run_duration, display_table=False
        )
        return EvaluationResult(
            test_results=test_results, confident_link=confident_link
        )

    elif test_cases and metrics:
        check_valid_test_cases_type(test_cases)

        global_test_run_manager.reset()
        start_time = time.perf_counter()

        if display_config.show_indicator:
            console = Console()
            for metric in metrics:
                console.print(
                    format_metric_description(
                        metric, async_mode=async_config.run_async
                    )
                )

        with capture_evaluation_run("evaluate()"):
            if async_config.run_async:
                loop = get_or_create_event_loop()
                test_results = loop.run_until_complete(
                    a_execute_test_cases(
                        test_cases,
                        metrics,
                        identifier=identifier,
                        ignore_errors=error_config.ignore_errors,
                        skip_on_missing_params=error_config.skip_on_missing_params,
                        use_cache=cache_config.use_cache,
                        save_to_disk=cache_config.write_cache,
                        verbose_mode=display_config.verbose_mode,
                        show_indicator=display_config.show_indicator,
                        throttle_value=async_config.throttle_value,
                        max_concurrent=async_config.max_concurrent,
                    )
                )
            else:
                test_results = execute_test_cases(
                    test_cases,
                    metrics,
                    identifier=identifier,
                    ignore_errors=error_config.ignore_errors,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                    use_cache=cache_config.use_cache,
                    save_to_disk=cache_config.write_cache,
                    show_indicator=display_config.show_indicator,
                    verbose_mode=display_config.verbose_mode,
                )

        end_time = time.perf_counter()
        run_duration = end_time - start_time
        if display_config.print_results:
            for test_result in test_results:
                print_test_result(test_result, display_config.display_option)
                aggregate_metric_pass_rates(test_results)
        if display_config.file_output_dir is not None:
            for test_result in test_results:
                write_test_result_to_file(
                    test_result,
                    display_config.display_option,
                    display_config.file_output_dir,
                )

        test_run = global_test_run_manager.get_test_run()
        test_run.hyperparameters = process_hyperparameters(hyperparameters)
        global_test_run_manager.save_test_run(TEMP_FILE_PATH)
        confident_link = global_test_run_manager.wrap_up_test_run(
            run_duration, display_table=False
        )
        return EvaluationResult(
            test_results=test_results, confident_link=confident_link
        )
