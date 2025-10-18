from typing import (
    Callable,
    List,
    Optional,
    Union,
    Dict,
    Any,
    Awaitable,
)
from rich.console import Console
import time

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.evaluate.api import APIEvaluate
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
from deepeval.test_run.hyperparameters import (
    process_hyperparameters,
    process_prompts,
)
from deepeval.test_run.test_run import TEMP_FILE_PATH
from deepeval.utils import (
    get_or_create_event_loop,
    open_browser,
    should_ignore_errors,
    should_skip_on_missing_params,
    should_use_cache,
    should_verbose_print,
    get_identifier,
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

    async_config = AsyncConfig(throttle_value=0, max_concurrent=100)
    display_config = DisplayConfig(
        verbose_mode=should_verbose_print(), show_indicator=True
    )
    error_config = ErrorConfig(
        ignore_errors=should_ignore_errors(),
        skip_on_missing_params=should_skip_on_missing_params(),
    )
    cache_config = CacheConfig(
        write_cache=get_is_running_deepeval(), use_cache=should_use_cache()
    )

    if golden and observed_callback:
        if run_async:
            loop = get_or_create_event_loop()
            test_result = loop.run_until_complete(
                a_execute_agentic_test_cases(
                    goldens=[golden],
                    observed_callback=observed_callback,
                    error_config=error_config,
                    display_config=display_config,
                    async_config=async_config,
                    cache_config=cache_config,
                    identifier=get_identifier(),
                    _use_bar_indicator=True,
                    _is_assert_test=True,
                )
            )[0]
        else:
            test_result = execute_agentic_test_cases(
                goldens=[golden],
                observed_callback=observed_callback,
                error_config=error_config,
                display_config=display_config,
                cache_config=cache_config,
                identifier=get_identifier(),
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
                    error_config=error_config,
                    display_config=display_config,
                    async_config=async_config,
                    cache_config=cache_config,
                    identifier=get_identifier(),
                    _use_bar_indicator=True,
                    _is_assert_test=True,
                )
            )[0]
        else:
            test_result = execute_test_cases(
                [test_case],
                metrics,
                error_config=error_config,
                display_config=display_config,
                cache_config=cache_config,
                identifier=get_identifier(),
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
    test_cases: Union[
        List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
    ],
    metrics: Optional[
        Union[
            List[BaseMetric],
            List[BaseConversationalMetric],
            List[BaseMultimodalMetric],
        ]
    ] = None,
    # Evals on Confident AI
    metric_collection: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Union[str, int, float, Prompt]]] = None,
    # agnostic
    identifier: Optional[str] = None,
    # Configs
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
) -> EvaluationResult:
    validate_evaluate_inputs(
        test_cases=test_cases,
        metrics=metrics,
        metric_collection=metric_collection,
    )
    check_valid_test_cases_type(test_cases)

    if metrics:

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
                        error_config=error_config,
                        display_config=display_config,
                        cache_config=cache_config,
                        async_config=async_config,
                    )
                )
            else:
                test_results = execute_test_cases(
                    test_cases,
                    metrics,
                    identifier=identifier,
                    error_config=error_config,
                    display_config=display_config,
                    cache_config=cache_config,
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
        test_run.prompts = process_prompts(hyperparameters)
        global_test_run_manager.save_test_run(TEMP_FILE_PATH)
        res = global_test_run_manager.wrap_up_test_run(
            run_duration, display_table=False
        )
        if isinstance(res, tuple):
            confident_link, test_run_id = res
        else:
            confident_link = test_run_id = None
        return EvaluationResult(
            test_results=test_results,
            confident_link=confident_link,
            test_run_id=test_run_id,
        )
    elif metric_collection:
        api = Api()
        api_evaluate = APIEvaluate(
            metricCollection=metric_collection,
            llmTestCases=(
                test_cases if isinstance(test_cases[0], LLMTestCase) else None
            ),
            conversationalTestCases=(
                test_cases
                if isinstance(test_cases[0], ConversationalTestCase)
                else None
            ),
        )
        try:
            body = api_evaluate.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = api_evaluate.dict(by_alias=True, exclude_none=True)

        _, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.EVALUATE_ENDPOINT,
            body=body,
        )
        if link:
            console = Console()
            console.print(
                "✅ Evaluation successfully pushed to Confident AI! View at "
                f"[link={link}]{link}[/link]"
            )
            open_browser(link)
