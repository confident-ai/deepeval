from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from typing import (
    Callable,
    List,
    Optional,
    Union,
    Any,
    Awaitable,
    Iterator,
)
from copy import deepcopy
import inspect
import asyncio
import time
import ast

from deepeval.tracing.tracing import (
    Observer,
    trace_manager,
    Trace,
    BaseSpan,
    AgentSpan,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
    perf_counter_to_datetime,
    to_zod_compatible_iso,
)
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.api import (
    TraceApi,
    BaseApiSpan,
)
from deepeval.dataset import Golden
from deepeval.dataset.types import global_evaluation_tasks
from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.utils import copy_metrics
from deepeval.utils import (
    get_or_create_event_loop,
)
from deepeval.telemetry import capture_evaluation_run
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
    TaskCompletionMetric,
)
from deepeval.metrics.indicator import (
    measure_metrics_with_indicator,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
)
from deepeval.test_run import (
    global_test_run_manager,
    LLMApiTestCase,
    ConversationalApiTestCase,
    TestRunManager,
    TestRun,
)
from deepeval.test_run.cache import (
    global_test_run_cache_manager,
    Cache,
    CachedTestCase,
    CachedMetricData,
)
from deepeval.evaluate.types import TestResult
from deepeval.evaluate.utils import (
    create_metric_data,
    create_test_result,
    create_api_test_case,
    count_metrics_in_trace,
    extract_trace_test_results,
)
from deepeval.utils import add_pbar, update_pbar, custom_console
from deepeval.openai.utils import openai_test_case_pairs
from deepeval.test_run.hyperparameters import process_hyperparameters
from deepeval.tracing.types import TestCaseMetricPair


def execute_test_cases(
    test_cases: Union[
        List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
    ],
    metrics: Union[
        List[BaseMetric],
        List[BaseConversationalMetric],
        List[BaseMultimodalMetric],
    ],
    skip_on_missing_params: bool,
    ignore_errors: bool,
    use_cache: bool,
    show_indicator: bool,
    save_to_disk: bool = False,
    verbose_mode: Optional[bool] = None,
    identifier: Optional[str] = None,
    test_run_manager: Optional[TestRunManager] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:
    global_test_run_cache_manager.disable_write_cache = save_to_disk == False

    if test_run_manager is None:
        test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run(identifier=identifier)

    if verbose_mode is not None:
        for metric in metrics:
            metric.verbose_mode = verbose_mode

    conversational_metrics: List[BaseConversationalMetric] = []
    llm_metrics: List[BaseMetric] = []
    mllm_metrics: List[BaseMultimodalMetric] = []
    for metric in metrics:
        metric.async_mode = False
        if isinstance(metric, BaseMetric):
            llm_metrics.append(metric)
        elif isinstance(metric, BaseConversationalMetric):
            conversational_metrics.append(metric)
        elif isinstance(metric, BaseMultimodalMetric):
            mllm_metrics.append(metric)

    test_results: List[TestResult] = []

    def evaluate_test_cases(
        progress: Optional[Progress] = None, pbar_id: Optional[str] = None
    ):
        llm_test_case_count = -1
        conversational_test_case_count = -1
        show_metric_indicator = show_indicator and not _use_bar_indicator
        for i, test_case in enumerate(test_cases):
            pbar_test_case_id = add_pbar(
                progress,
                f"    🎯 Evaluating test case #{i}",
                total=len(metrics),
            )
            with capture_evaluation_run("test case"):
                for metric in metrics:
                    metric.error = None  # Reset metric error

                if isinstance(test_case, LLMTestCase):
                    if len(llm_metrics) == 0:
                        continue

                    llm_test_case_count += 1
                    cached_test_case = None
                    if use_cache:
                        cached_test_case = (
                            global_test_run_cache_manager.get_cached_test_case(
                                test_case, test_run.hyperparameters
                            )
                        )

                    ##### Metric Calculation #####
                    api_test_case: LLMApiTestCase = create_api_test_case(
                        test_case=test_case, index=llm_test_case_count
                    )
                    new_cached_test_case: CachedTestCase = CachedTestCase()

                    test_start_time = time.perf_counter()
                    read_all_metrics_from_cache = True
                    for metric in llm_metrics:
                        metric_data = None
                        if cached_test_case is not None:
                            cached_metric_data = Cache.get_metric_data(
                                metric, cached_test_case
                            )
                            if cached_metric_data:
                                metric_data = cached_metric_data.metric_data

                        if metric_data is None:
                            read_all_metrics_from_cache = False
                            try:
                                metric.measure(
                                    test_case,
                                    _show_indicator=show_metric_indicator,
                                )
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except TypeError:
                                try:
                                    metric.measure(test_case)
                                except MissingTestCaseParamsError as e:
                                    if skip_on_missing_params:
                                        continue
                                    else:
                                        if ignore_errors:
                                            metric.error = str(e)
                                            metric.success = False
                                        else:
                                            raise
                                except Exception as e:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                            metric_data = create_metric_data(metric)

                        # here, we will check for an additional property on the flattened test cases to see if updating is necessary
                        api_test_case.update_metric_data(metric_data)
                        if metric.error is None:
                            cache_metric_data = deepcopy(metric_data)
                            cache_metric_data.evaluation_cost = 0  # Cached metrics will have evaluation cost as 0, not None.
                            updated_cached_metric_data = CachedMetricData(
                                metric_data=cache_metric_data,
                                metric_configuration=Cache.create_metric_configuration(
                                    metric
                                ),
                            )
                            new_cached_test_case.cached_metrics_data.append(
                                updated_cached_metric_data
                            )
                        update_pbar(progress, pbar_test_case_id)

                    test_end_time = time.perf_counter()
                    if read_all_metrics_from_cache:
                        run_duration = 0
                    else:
                        run_duration = test_end_time - test_start_time
                    api_test_case.update_run_duration(run_duration)

                    ### Update Test Run ###
                    test_run_manager.update_test_run(api_test_case, test_case)

                    ### Cache Test Run ###
                    global_test_run_cache_manager.cache_test_case(
                        test_case,
                        new_cached_test_case,
                        test_run.hyperparameters,
                    )
                    global_test_run_cache_manager.cache_test_case(
                        test_case,
                        new_cached_test_case,
                        test_run.hyperparameters,
                        to_temp=True,
                    )

                # No caching and not sending test cases to Confident AI for multimodal metrics yet
                elif isinstance(test_case, MLLMTestCase):
                    if len(mllm_metrics) == 0:
                        continue

                    api_test_case: LLMApiTestCase = create_api_test_case(
                        test_case=test_case, index=llm_test_case_count
                    )
                    test_start_time = time.perf_counter()
                    for metric in mllm_metrics:
                        try:
                            metric.measure(
                                test_case,
                                _show_indicator=show_metric_indicator,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(test_case)
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise
                        metric_data = create_metric_data(metric)
                        api_test_case.update_metric_data(metric_data)
                        update_pbar(progress, pbar_test_case_id)

                    test_end_time = time.perf_counter()
                    if len(mllm_metrics) > 0:
                        run_duration = test_end_time - test_start_time
                        api_test_case.update_run_duration(run_duration)

                    ### Update Test Run ###
                    test_run_manager.update_test_run(api_test_case, test_case)

                # No caching for conversational metrics yet
                elif isinstance(test_case, ConversationalTestCase):
                    if len(metrics) == 0:
                        continue

                    conversational_test_case_count += 1
                    api_test_case: ConversationalApiTestCase = (
                        create_api_test_case(
                            test_case=test_case,
                            index=conversational_test_case_count,
                        )
                    )

                    test_start_time = time.perf_counter()
                    for metric in metrics:
                        try:
                            metric.measure(
                                test_case,
                                _show_indicator=show_metric_indicator,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(test_case)
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise
                        metric_data = create_metric_data(metric)
                        api_test_case.update_metric_data(metric_data)
                        update_pbar(progress, pbar_test_case_id)

                    test_end_time = time.perf_counter()
                    run_duration = test_end_time - test_start_time
                    api_test_case.update_run_duration(run_duration)

                    ### Update Test Run ###
                    test_run_manager.update_test_run(api_test_case, test_case)

                test_result = create_test_result(api_test_case)
                test_results.append(test_result)
                update_pbar(progress, pbar_id)

    if show_indicator and _use_bar_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        with progress:
            pbar_id = add_pbar(
                progress,
                f"Evaluating {len(test_cases)} test case(s) sequentially",
                total=len(test_cases),
            )
            evaluate_test_cases(progress=progress, pbar_id=pbar_id)
    else:
        evaluate_test_cases()

    return test_results


async def a_execute_test_cases(
    test_cases: Union[
        List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
    ],
    metrics: Union[
        List[BaseMetric],
        List[BaseConversationalMetric],
        List[BaseMultimodalMetric],
    ],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    use_cache: bool,
    show_indicator: bool,
    throttle_value: int,
    max_concurrent: int,
    save_to_disk: bool = False,
    verbose_mode: Optional[bool] = None,
    identifier: Optional[str] = None,
    test_run_manager: Optional[TestRunManager] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    global_test_run_cache_manager.disable_write_cache = save_to_disk == False
    if test_run_manager is None:
        test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run(identifier=identifier)

    if verbose_mode is not None:
        for metric in metrics:
            metric.verbose_mode = verbose_mode

    llm_metrics: List[BaseMetric] = []
    mllm_metrics: List[BaseMultimodalMetric] = []
    conversational_metrics: List[BaseConversationalMetric] = []
    for metric in metrics:
        if isinstance(metric, BaseMetric):
            llm_metrics.append(metric)
        elif isinstance(metric, BaseMultimodalMetric):
            mllm_metrics.append(metric)
        elif isinstance(metric, BaseConversationalMetric):
            conversational_metrics.append(metric)

    llm_test_case_counter = -1
    mllm_test_case_counter = -1
    conversational_test_case_counter = -1
    test_results: List[Union[TestResult, MLLMTestCase]] = []
    tasks = []

    if show_indicator and _use_bar_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        pbar_id = add_pbar(
            progress,
            f"Evaluating {len(test_cases)} test case(s) in parallel",
            total=len(test_cases),
        )
        with progress:
            for test_case in test_cases:
                with capture_evaluation_run("test case"):
                    if isinstance(test_case, LLMTestCase):
                        if len(llm_metrics) == 0:
                            update_pbar(progress, pbar_id)
                            continue

                        llm_test_case_counter += 1
                        copied_llm_metrics: List[BaseMetric] = copy_metrics(
                            llm_metrics
                        )
                        task = execute_with_semaphore(
                            func=a_execute_llm_test_cases,
                            metrics=copied_llm_metrics,
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=llm_test_case_counter,
                            test_run=test_run,
                            ignore_errors=ignore_errors,
                            skip_on_missing_params=skip_on_missing_params,
                            use_cache=use_cache,
                            show_indicator=show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            _is_assert_test=_is_assert_test,
                            progress=progress,
                            pbar_id=pbar_id,
                        )
                        tasks.append(asyncio.create_task(task))

                    elif isinstance(test_case, MLLMTestCase):
                        mllm_test_case_counter += 1
                        copied_multimodal_metrics: List[
                            BaseMultimodalMetric
                        ] = copy_metrics(mllm_metrics)
                        task = execute_with_semaphore(
                            func=a_execute_mllm_test_cases,
                            metrics=copied_multimodal_metrics,
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=mllm_test_case_counter,
                            ignore_errors=ignore_errors,
                            skip_on_missing_params=skip_on_missing_params,
                            show_indicator=show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            _is_assert_test=_is_assert_test,
                            progress=progress,
                            pbar_id=pbar_id,
                        )
                        tasks.append(asyncio.create_task(task))

                    elif isinstance(test_case, ConversationalTestCase):
                        conversational_test_case_counter += 1

                        task = execute_with_semaphore(
                            func=a_execute_conversational_test_cases,
                            metrics=copy_metrics(metrics),
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=conversational_test_case_counter,
                            ignore_errors=ignore_errors,
                            skip_on_missing_params=skip_on_missing_params,
                            show_indicator=show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            _is_assert_test=_is_assert_test,
                            progress=progress,
                            pbar_id=pbar_id,
                        )
                        tasks.append(asyncio.create_task(task))

                    await asyncio.sleep(throttle_value)
            await asyncio.gather(*tasks)
    else:
        for test_case in test_cases:
            with capture_evaluation_run("test case"):
                if isinstance(test_case, LLMTestCase):
                    if len(llm_metrics) == 0:
                        continue
                    llm_test_case_counter += 1

                    copied_llm_metrics: List[BaseMetric] = copy_metrics(
                        llm_metrics
                    )
                    task = execute_with_semaphore(
                        func=a_execute_llm_test_cases,
                        metrics=copied_llm_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=llm_test_case_counter,
                        test_run=test_run,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        use_cache=use_cache,
                        _use_bar_indicator=_use_bar_indicator,
                        _is_assert_test=_is_assert_test,
                        show_indicator=show_indicator,
                    )
                    tasks.append(asyncio.create_task((task)))

                elif isinstance(test_case, ConversationalTestCase):
                    conversational_test_case_counter += 1
                    copied_conversational_metrics: List[
                        BaseConversationalMetric
                    ] = []
                    copied_conversational_metrics = copy_metrics(
                        conversational_metrics
                    )
                    task = execute_with_semaphore(
                        func=a_execute_conversational_test_cases,
                        metrics=copied_conversational_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=conversational_test_case_counter,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        _use_bar_indicator=_use_bar_indicator,
                        _is_assert_test=_is_assert_test,
                        show_indicator=show_indicator,
                    )
                    tasks.append(asyncio.create_task((task)))

                elif isinstance(test_case, MLLMTestCase):
                    mllm_test_case_counter += 1
                    copied_multimodal_metrics: List[BaseMultimodalMetric] = (
                        copy_metrics(mllm_metrics)
                    )
                    task = execute_with_semaphore(
                        func=a_execute_mllm_test_cases,
                        metrics=copied_multimodal_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=mllm_test_case_counter,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        _use_bar_indicator=_use_bar_indicator,
                        _is_assert_test=_is_assert_test,
                        show_indicator=show_indicator,
                    )
                    tasks.append(asyncio.create_task(task))

                await asyncio.sleep(throttle_value)
        await asyncio.gather(*tasks)

    return test_results


async def a_execute_llm_test_cases(
    metrics: List[BaseMetric],
    test_case: LLMTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    test_run: TestRun,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    use_cache: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
    pbar_test_case_id = add_pbar(
        progress,
        f"    🎯 Evaluating test case #{count}",
        total=len(metrics),
    )
    show_metrics_indicator = show_indicator and not _use_bar_indicator

    cached_test_case = None
    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error

    # only use cache when NOT conversational test case
    if use_cache:
        cached_test_case = global_test_run_cache_manager.get_cached_test_case(
            test_case,
            test_run.hyperparameters,
        )

    ##### Metric Calculation #####
    api_test_case = create_api_test_case(
        test_case=test_case, index=count if not _is_assert_test else None
    )
    new_cached_test_case: CachedTestCase = CachedTestCase()
    test_start_time = time.perf_counter()
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=cached_test_case,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
        pbar_eval_id=pbar_test_case_id,
        progress=progress,
    )

    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        api_test_case.update_metric_data(metric_data)

        if metric.error is None:
            cache_metric_data = deepcopy(metric_data)
            cache_metric_data.evaluation_cost = (
                0  # Create new copy and save 0 for cost
            )
            updated_cached_metric_data = CachedMetricData(
                metric_data=cache_metric_data,
                metric_configuration=Cache.create_metric_configuration(metric),
            )
            new_cached_test_case.cached_metrics_data.append(
                updated_cached_metric_data
            )

    test_end_time = time.perf_counter()
    run_duration = test_end_time - test_start_time
    # Quick hack to check if all metrics were from cache
    if run_duration < 1:
        run_duration = 0
    api_test_case.update_run_duration(run_duration)

    ### Update Test Run ###
    test_run_manager.update_test_run(api_test_case, test_case)

    ### Cache Test Run ###
    global_test_run_cache_manager.cache_test_case(
        test_case,
        new_cached_test_case,
        test_run.hyperparameters,
    )
    global_test_run_cache_manager.cache_test_case(
        test_case,
        new_cached_test_case,
        test_run.hyperparameters,
        to_temp=True,
    )

    test_results.append(create_test_result(api_test_case))
    update_pbar(progress, pbar_id)


async def a_execute_mllm_test_cases(
    metrics: List[BaseMultimodalMetric],
    test_case: MLLMTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
    show_metrics_indicator = show_indicator and not _use_bar_indicator
    pbar_test_case_id = add_pbar(
        progress,
        f"    🎯 Evaluating test case #{count}",
        total=len(metrics),
    )

    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error

    api_test_case: LLMApiTestCase = create_api_test_case(
        test_case=test_case, index=count if not _is_assert_test else None
    )
    test_start_time = time.perf_counter()
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=None,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
        pbar_eval_id=pbar_test_case_id,
        progress=progress,
    )
    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        api_test_case.update_metric_data(metric_data)

    test_end_time = time.perf_counter()
    run_duration = test_end_time - test_start_time
    api_test_case.update_run_duration(run_duration)

    ### Update Test Run ###
    test_run_manager.update_test_run(api_test_case, test_case)
    test_results.append(create_test_result(api_test_case))
    update_pbar(progress, pbar_id)


async def a_execute_conversational_test_cases(
    metrics: List[
        Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric]
    ],
    test_case: ConversationalTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
    show_metrics_indicator = show_indicator and not _use_bar_indicator
    pbar_test_case_id = add_pbar(
        progress,
        f"    🎯 Evaluating test case #{count}",
        total=len(metrics),
    )

    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error

    api_test_case: ConversationalApiTestCase = create_api_test_case(
        test_case=test_case, index=count if not _is_assert_test else None
    )

    test_start_time = time.perf_counter()
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=None,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
        pbar_eval_id=pbar_test_case_id,
        progress=progress,
    )
    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        api_test_case.update_metric_data(metric_data)

    test_end_time = time.perf_counter()
    if len(metrics) > 0:
        run_duration = test_end_time - test_start_time
        api_test_case.update_run_duration(run_duration)

    ### Update Test Run ###
    test_run_manager.update_test_run(api_test_case, test_case)

    test_results.append(create_test_result(api_test_case))
    update_pbar(progress, pbar_id)


###########################################
### Component-Level Evals
###########################################


def execute_agentic_test_cases(
    goldens: List[Golden],
    observed_callback: Union[
        Callable[[str], Any], Callable[[str], Awaitable[Any]]
    ],
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    save_to_disk: bool = False,
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:

    test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = save_to_disk
    test_run_manager.get_test_run(identifier=identifier)

    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True
    test_results: List[TestResult] = []

    def evaluate_test_cases(
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ):
        count = 0
        show_metric_indicator = show_indicator and not _use_bar_indicator

        for golden in goldens:
            with capture_evaluation_run("golden"):
                count += 1
                total_tags = count_observe_decorators_in_module(
                    observed_callback
                )
                pbar_tags_id = add_pbar(
                    progress,
                    f"     ⚡ Invoking observed callback (#{count})",
                    total=total_tags,
                )

                with Observer(
                    "custom",
                    func_name="Test Wrapper",
                    _progress=progress,
                    _pbar_callback_id=pbar_tags_id,
                ):
                    if asyncio.iscoroutinefunction(observed_callback):
                        loop = get_or_create_event_loop()
                        loop.run_until_complete(observed_callback(golden.input))
                    else:
                        observed_callback(golden.input)
                    current_trace: Trace = current_trace_context.get()

                update_pbar(progress, pbar_tags_id, advance=total_tags)
                update_pbar(progress, pbar_id)

                # Create empty trace api for llm api test case
                trace_api = TraceApi(
                    uuid=current_trace.uuid,
                    baseSpans=[],
                    agentSpans=[],
                    llmSpans=[],
                    retrieverSpans=[],
                    toolSpans=[],
                    startTime=(
                        to_zod_compatible_iso(
                            perf_counter_to_datetime(current_trace.start_time)
                        )
                        if current_trace.start_time
                        else None
                    ),
                    endTime=(
                        to_zod_compatible_iso(
                            perf_counter_to_datetime(current_trace.end_time)
                        )
                        if current_trace.end_time
                        else None
                    ),
                )

                # Format golden as test case to create llm api test case
                test_case = LLMTestCase(
                    input=golden.input,
                    actual_output=(
                        str(current_trace.output)
                        if current_trace.output is not None
                        else None
                    ),
                    expected_output=current_trace.expected_output,
                    context=current_trace.context,
                    retrieval_context=current_trace.retrieval_context,
                    additional_metadata=golden.additional_metadata,
                    tools_called=current_trace.tools_called,
                    expected_tools=current_trace.expected_tools,
                    comments=golden.comments,
                    name=golden.name,
                    _dataset_alias=golden._dataset_alias,
                    _dataset_id=golden._dataset_id,
                )
                api_test_case = create_api_test_case(
                    test_case=test_case,
                    trace=trace_api,
                    index=count if not _is_assert_test else None,
                )

                # Run DFS to calculate metrics synchronously
                def dfs(
                    span: BaseSpan,
                    progress: Optional[Progress] = None,
                    pbar_eval_id: Optional[int] = None,
                ):
                    # Create API Span
                    metrics: List[BaseMetric] = span.metrics
                    api_span: BaseApiSpan = (
                        trace_manager._convert_span_to_api_span(span)
                    )
                    if isinstance(span, AgentSpan):
                        trace_api.agent_spans.append(api_span)
                    elif isinstance(span, LlmSpan):
                        trace_api.llm_spans.append(api_span)
                    elif isinstance(span, RetrieverSpan):
                        trace_api.retriever_spans.append(api_span)
                    elif isinstance(span, ToolSpan):
                        trace_api.tool_spans.append(api_span)
                    else:
                        trace_api.base_spans.append(api_span)

                    for child in span.children:
                        dfs(child, progress, pbar_eval_id)

                    if span.metrics is None:
                        return
                    has_task_completion = any(
                        isinstance(metric, TaskCompletionMetric)
                        for metric in span.metrics
                    )

                    llm_test_case = None
                    if span.input is None:
                        llm_test_case = LLMTestCase(
                            input=str(span.input),
                            actual_output=str(span.output),
                            expected_output=span.expected_output,
                            context=span.context,
                            retrieval_context=span.retrieval_context,
                            tools_called=span.tools_called,
                            expected_tools=span.expected_tools,
                        )
                    if llm_test_case is None and not has_task_completion:
                        raise ValueError(
                            "Unable to run metrics on span without LLMTestCase. Are you sure you called `update_current_span()`?"
                        )

                    # add trace if task completion
                    if has_task_completion:
                        if llm_test_case is None:
                            llm_test_case = LLMTestCase(
                                input="None", actual_output="None"
                            )
                        llm_test_case._trace_dict = (
                            trace_manager.create_nested_spans_dict(span)
                        )

                    # Preparing metric calculation
                    api_span.metrics_data = []
                    for metric in metrics:
                        metric.skipped = False
                        metric.error = None
                        if verbose_mode is not None:
                            metric.verbose_mode = verbose_mode

                    # Metric calculation
                    for metric in metrics:
                        metric_data = None
                        try:
                            metric.measure(
                                llm_test_case,
                                _show_indicator=show_metric_indicator,
                                _in_component=True,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(
                                    llm_test_case, _in_component=True
                                )
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise
                        metric_data = create_metric_data(metric)
                        api_span.metrics_data.append(metric_data)
                        api_test_case.update_status(metric_data.success)
                        update_pbar(progress, pbar_eval_id)

                trace_level_metrics_count = (
                    len(current_trace.metrics) if current_trace.metrics else 0
                )
                pbar_eval_id = add_pbar(
                    progress,
                    f"     🎯 Evaluating component(s) (#{count})",
                    total=count_metrics_in_trace(trace=current_trace)
                    + trace_level_metrics_count,
                )

                start_time = time.perf_counter()

                # Handle trace-level metrics
                if current_trace.metrics:
                    has_task_completion = any(
                        isinstance(metric, TaskCompletionMetric)
                        for metric in current_trace.metrics
                    )

                    llm_test_case = None
                    if current_trace.input:
                        llm_test_case = LLMTestCase(
                            input=str(current_trace.input),
                            actual_output=(
                                str(current_trace.output)
                                if current_trace.output is not None
                                else None
                            ),
                            expected_output=current_trace.expected_output,
                            context=current_trace.context,
                            retrieval_context=current_trace.retrieval_context,
                            tools_called=current_trace.tools_called,
                            expected_tools=current_trace.expected_tools,
                        )
                    if llm_test_case is None and not has_task_completion:
                        raise ValueError(
                            "Unable to run metrics on trace without LLMTestCase. Are you sure you called `update_current_trace()`?"
                        )

                    if has_task_completion:
                        if llm_test_case is None:
                            llm_test_case = LLMTestCase(
                                input="None", actual_output="None"
                            )
                        llm_test_case._trace_dict = (
                            trace_manager.create_nested_spans_dict(
                                current_trace.root_spans[0]
                            )
                        )

                    for metric in current_trace.metrics:
                        metric.skipped = False
                        metric.error = None
                        if verbose_mode is not None:
                            metric.verbose_mode = verbose_mode

                    trace_api.metrics_data = []
                    for metric in current_trace.metrics:
                        try:
                            metric.measure(
                                llm_test_case,
                                _show_indicator=show_metric_indicator,
                                _in_component=True,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(
                                    llm_test_case, _in_component=True
                                )
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise

                        if not metric.skipped:
                            metric_data = create_metric_data(metric)
                            trace_api.metrics_data.append(metric_data)
                            api_test_case.update_metric_data(metric_data)
                            api_test_case.update_status(metric_data.success)
                            update_pbar(progress, pbar_eval_id)

                # Then handle span-level metrics
                dfs(current_trace.root_spans[0], progress, pbar_eval_id)
                end_time = time.perf_counter()
                run_duration = end_time - start_time

                # Update test run
                api_test_case.update_run_duration(run_duration)
                test_run_manager.update_test_run(api_test_case, test_case)
                test_results.append(create_test_result(api_test_case))
                test_results.extend(extract_trace_test_results(trace_api))

                update_pbar(progress, pbar_id)

    if show_indicator and _use_bar_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        with progress:
            pbar_id = add_pbar(
                progress,
                f"Running Component-Level Evals (sync)",
                total=len(goldens) * 2,
            )
            evaluate_test_cases(progress=progress, pbar_id=pbar_id)
    else:
        evaluate_test_cases()

    local_trace_manager.evaluating = False
    return test_results


async def a_execute_agentic_test_cases(
    goldens: List[Golden],
    observed_callback: Union[
        Callable[[str], Any], Callable[[str], Awaitable[Any]]
    ],
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    throttle_value: int,
    max_concurrent: int,
    save_to_disk: bool = False,
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    test_run_manager = global_test_run_manager
    test_run_manager.save_to_disk = save_to_disk
    test_run_manager.get_test_run(identifier=identifier)
    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True
    test_results: List[TestResult] = []
    tasks = []
    count = 0

    if show_indicator and _use_bar_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        with progress:
            pbar_id = add_pbar(
                progress,
                "Running Component-Level Evals (async)",
                total=len(goldens) * 2,
            )
            for golden in goldens:
                with capture_evaluation_run("golden"):
                    count += 1
                    task = execute_with_semaphore(
                        func=a_execute_agentic_test_case,
                        golden=golden,
                        observed_callback=observed_callback,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=count,
                        verbose_mode=verbose_mode,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        show_indicator=show_indicator,
                        _use_bar_indicator=_use_bar_indicator,
                        _is_assert_test=_is_assert_test,
                        progress=progress,
                        pbar_id=pbar_id,
                    )
                    tasks.append(asyncio.create_task(task))
                    await asyncio.sleep(throttle_value)

            await asyncio.gather(*tasks)
    else:
        for golden in goldens:
            with capture_evaluation_run("golden"):
                count += 1
                task = execute_with_semaphore(
                    func=a_execute_agentic_test_case,
                    golden=golden,
                    observed_callback=observed_callback,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    count=count,
                    verbose_mode=verbose_mode,
                    ignore_errors=ignore_errors,
                    skip_on_missing_params=skip_on_missing_params,
                    show_indicator=show_indicator,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                )
                tasks.append(asyncio.create_task(task))
                await asyncio.sleep(throttle_value)
        await asyncio.gather(*tasks)
    local_trace_manager.evaluating = False
    return test_results


async def a_execute_agentic_test_case(
    golden: Golden,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    observed_callback: Optional[
        Union[Callable[[str], Any], Callable[[str], Awaitable[Any]]]
    ] = None,
    trace: Optional[Trace] = None,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
    if observed_callback:
        total_tags = count_observe_decorators_in_module(observed_callback)
        pbar_tags_id = add_pbar(
            progress,
            f"     ⚡ Invoking observed callback (#{count})",
            total=total_tags,
        )

        # Call callback and extract trace
        with Observer(
            "custom",
            func_name="Test Wrapper",
            _progress=progress,
            _pbar_callback_id=pbar_tags_id,
        ):
            if asyncio.iscoroutinefunction(observed_callback):
                await observed_callback(golden.input)
            else:
                observed_callback(golden.input)
            current_trace: Trace = current_trace_context.get()

        update_pbar(progress, pbar_tags_id, advance=total_tags)
        update_pbar(progress, pbar_id)
    elif trace:
        current_trace = trace

    # run evals through DFS
    trace_api = TraceApi(
        uuid=current_trace.uuid,
        baseSpans=[],
        agentSpans=[],
        llmSpans=[],
        retrieverSpans=[],
        toolSpans=[],
        startTime=(
            to_zod_compatible_iso(
                perf_counter_to_datetime(current_trace.start_time)
            )
            if current_trace.start_time
            else None
        ),
        endTime=(
            to_zod_compatible_iso(
                perf_counter_to_datetime(current_trace.end_time)
            )
            if current_trace.end_time
            else None
        ),
    )

    trace_level_metrics_count = (
        len(current_trace.metrics) if current_trace.metrics else 0
    )
    pbar_eval_id = add_pbar(
        progress,
        f"     🎯 Evaluating component(s) (#{count})",
        total=count_metrics_in_trace(trace=current_trace)
        + trace_level_metrics_count,
    )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(trace.output) if trace.output is not None else None,
        expected_output=trace.expected_output,
        context=trace.context,
        retrieval_context=trace.retrieval_context,
        tools_called=trace.tools_called,
        expected_tools=trace.expected_tools,
        additional_metadata=golden.additional_metadata,
        comments=golden.comments,
        name=golden.name,
        _dataset_alias=golden._dataset_alias,
        _dataset_id=golden._dataset_id,
    )
    api_test_case = create_api_test_case(
        test_case=test_case,
        trace=trace_api,
        index=count if not _is_assert_test else None,
    )

    await a_execute_trace_test_case(
        trace=trace,
        trace_api=trace_api,
        api_test_case=api_test_case,
        ignore_errors=ignore_errors,
        skip_on_missing_params=skip_on_missing_params,
        show_indicator=show_indicator,
        verbose_mode=verbose_mode,
        progress=progress,
        pbar_eval_id=pbar_eval_id,
        _use_bar_indicator=_use_bar_indicator,
    )

    async def dfs(span: BaseSpan):
        await a_execute_span_test_case(
            span=span,
            trace_api=trace_api,
            api_test_case=api_test_case,
            ignore_errors=ignore_errors,
            skip_on_missing_params=skip_on_missing_params,
            show_indicator=show_indicator,
            verbose_mode=verbose_mode,
            progress=progress,
            pbar_eval_id=pbar_eval_id,
            _use_bar_indicator=_use_bar_indicator,
        )
        child_tasks = [dfs(child) for child in span.children]
        if child_tasks:
            await asyncio.gather(*child_tasks)

    test_start_time = time.perf_counter()
    await dfs(current_trace.root_spans[0])
    test_end_time = time.perf_counter()
    run_duration = test_end_time - test_start_time

    api_test_case.update_run_duration(run_duration)
    test_run_manager.update_test_run(api_test_case, test_case)
    test_results.append(create_test_result(api_test_case))
    test_results.extend(extract_trace_test_results(trace_api))

    update_pbar(progress, pbar_id)


async def a_execute_span_test_case(
    span: BaseSpan,
    trace_api: TraceApi,
    api_test_case: LLMApiTestCase,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    verbose_mode: Optional[bool],
    progress: Optional[Progress],
    pbar_eval_id: Optional[int],
    _use_bar_indicator: bool,
):
    api_span: BaseApiSpan = trace_manager._convert_span_to_api_span(span)
    if isinstance(span, AgentSpan):
        trace_api.agent_spans.append(api_span)
    elif isinstance(span, LlmSpan):
        trace_api.llm_spans.append(api_span)
    elif isinstance(span, RetrieverSpan):
        trace_api.retriever_spans.append(api_span)
    elif isinstance(span, ToolSpan):
        trace_api.tool_spans.append(api_span)
    else:
        trace_api.base_spans.append(api_span)

    if span.metrics is None:
        return

    has_task_completion = any(
        isinstance(metric, TaskCompletionMetric) for metric in span.metrics
    )

    llm_test_case = None
    if span.input:
        llm_test_case = LLMTestCase(
            input=str(span.input),
            actual_output=str(span.output),
            expected_output=span.expected_output,
            context=span.context,
            retrieval_context=span.retrieval_context,
            tools_called=span.tools_called,
            expected_tools=span.expected_tools,
        )
    if llm_test_case is None and not has_task_completion:
        raise ValueError(
            "Unable to run metrics on span without LLMTestCase. Are you sure you called `update_current_span()`?"
        )

    show_metrics_indicator = show_indicator and not _use_bar_indicator
    metrics: List[BaseMetric] = span.metrics
    test_case: Optional[LLMTestCase] = llm_test_case

    # add trace if task completion
    if has_task_completion:
        if test_case is None:
            test_case = LLMTestCase(input="None", actual_output="None")
        test_case._trace_dict = trace_manager.create_nested_spans_dict(span)

    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error
        if verbose_mode is not None:
            metric.verbose_mode = verbose_mode

    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=None,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
        progress=progress,
        pbar_eval_id=pbar_eval_id,
        _in_component=True,
    )

    api_span.metrics_data = []
    for metric in metrics:
        if metric.skipped:
            continue
        metric_data = create_metric_data(metric)
        api_span.metrics_data.append(metric_data)
        api_test_case.update_status(metric_data.success)


async def a_execute_trace_test_case(
    trace: Trace,
    trace_api: TraceApi,
    api_test_case: LLMApiTestCase,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    verbose_mode: Optional[bool],
    progress: Optional[Progress],
    pbar_eval_id: Optional[int],
    _use_bar_indicator: bool,
):
    if trace.metrics is None:
        return

    has_task_completion = any(
        isinstance(metric, TaskCompletionMetric) for metric in trace.metrics
    )

    llm_test_case = None
    if trace.input:
        llm_test_case = LLMTestCase(
            input=str(trace.input),
            actual_output=(
                str(trace.output) if trace.output is not None else None
            ),
            expected_output=trace.expected_output,
            context=trace.context,
            retrieval_context=trace.retrieval_context,
            tools_called=trace.tools_called,
            expected_tools=trace.expected_tools,
        )
    if llm_test_case is None and not has_task_completion:
        raise ValueError(
            "Unable to run metrics on trace without LLMTestCase. Are you sure you called `update_current_trace()`?"
        )

    show_metrics_indicator = show_indicator and not _use_bar_indicator
    metrics: List[BaseMetric] = trace.metrics
    test_case: Optional[LLMTestCase] = llm_test_case

    # add trace if task completion
    if has_task_completion:
        if test_case is None:
            test_case = LLMTestCase(input="None", actual_output="None")
        test_case._trace_dict = trace_manager.create_nested_spans_dict(
            trace.root_spans[0]
        )

    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error
        if verbose_mode is not None:
            metric.verbose_mode = verbose_mode

    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=None,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
        progress=progress,
        pbar_eval_id=pbar_eval_id,
        _in_component=True,
    )

    trace_api.metrics_data = []
    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        trace_api.metrics_data.append(metric_data)
        api_test_case.update_metric_data(metric_data)
        api_test_case.update_status(metric_data.success)


def count_observe_decorators_in_module(func: Callable) -> int:
    mod = inspect.getmodule(func)
    if mod is None or not hasattr(mod, "__file__"):
        raise RuntimeError("Cannot locate @observe function.")
    module_source = inspect.getsource(mod)
    tree = ast.parse(module_source)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Call)
                    and getattr(deco.func, "id", "") == "observe"
                ):
                    count += 1
    return count


###########################################
### Looped Evals
###########################################


def execute_agentic_test_cases_from_loop(
    goldens: List[Golden],
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    test_results: List[TestResult],
    save_to_disk: bool = False,
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> Iterator[TestResult]:

    test_run_manager = global_test_run_manager
    test_run_manager.save_to_disk = save_to_disk
    test_run_manager.get_test_run(identifier=identifier)

    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True

    def evaluate_test_cases(
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> Iterator[Golden]:
        count = 0
        show_metric_indicator = show_indicator and not _use_bar_indicator

        for golden in goldens:
            with capture_evaluation_run("golden"):
                # yield golden
                count += 1
                pbar_tags_id = add_pbar(
                    progress, f"\t⚡ Invoking observed callback (#{count})"
                )
                with Observer(
                    "custom",
                    func_name="Test Wrapper",
                    _progress=progress,
                    _pbar_callback_id=pbar_tags_id,
                ):
                    yield golden
                    current_trace: Trace = current_trace_context.get()

                update_pbar(progress, pbar_tags_id)
                update_pbar(progress, pbar_id)

                # Create empty trace api for llm api test case
                trace_api = TraceApi(
                    uuid=current_trace.uuid,
                    baseSpans=[],
                    agentSpans=[],
                    llmSpans=[],
                    retrieverSpans=[],
                    toolSpans=[],
                    startTime=(
                        to_zod_compatible_iso(
                            perf_counter_to_datetime(current_trace.start_time)
                        )
                        if current_trace.start_time
                        else None
                    ),
                    endTime=(
                        to_zod_compatible_iso(
                            perf_counter_to_datetime(current_trace.end_time)
                        )
                        if current_trace.end_time
                        else None
                    ),
                )

                # Format golden as test case to create llm api test case
                test_case = LLMTestCase(
                    input=golden.input,
                    actual_output=(
                        str(current_trace.output)
                        if current_trace.output is not None
                        else None
                    ),
                    expected_output=current_trace.expected_output,
                    context=current_trace.context,
                    retrieval_context=current_trace.retrieval_context,
                    additional_metadata=golden.additional_metadata,
                    tools_called=current_trace.tools_called,
                    expected_tools=current_trace.expected_tools,
                    comments=golden.comments,
                    name=golden.name,
                    _dataset_alias=golden._dataset_alias,
                    _dataset_id=golden._dataset_id,
                )
                api_test_case = create_api_test_case(
                    test_case=test_case,
                    trace=trace_api,
                    index=count if not _is_assert_test else None,
                )

                # Run DFS to calculate metrics synchronously
                def dfs(
                    span: BaseSpan,
                    progress: Optional[Progress] = None,
                    pbar_eval_id: Optional[int] = None,
                ):
                    # Create API Span
                    metrics: List[BaseMetric] = span.metrics
                    api_span: BaseApiSpan = (
                        trace_manager._convert_span_to_api_span(span)
                    )
                    if isinstance(span, AgentSpan):
                        trace_api.agent_spans.append(api_span)
                    elif isinstance(span, LlmSpan):
                        trace_api.llm_spans.append(api_span)
                    elif isinstance(span, RetrieverSpan):
                        trace_api.retriever_spans.append(api_span)
                    elif isinstance(span, ToolSpan):
                        trace_api.tool_spans.append(api_span)
                    else:
                        trace_api.base_spans.append(api_span)

                    for child in span.children:
                        dfs(child, progress, pbar_eval_id)

                    llm_test_case = None
                    if span.input is None:
                        llm_test_case = LLMTestCase(
                            input=str(span.input),
                            actual_output=str(span.output),
                            expected_output=span.expected_output,
                            context=span.context,
                            retrieval_context=span.retrieval_context,
                            tools_called=span.tools_called,
                            expected_tools=span.expected_tools,
                        )
                    if span.metrics == None or llm_test_case == None:
                        return

                    # Preparing metric calculation
                    api_span.metrics_data = []
                    for metric in metrics:
                        metric.skipped = False
                        metric.error = None
                        if verbose_mode is not None:
                            metric.verbose_mode = verbose_mode

                    # Metric calculation
                    for metric in metrics:
                        metric_data = None
                        try:
                            metric.measure(
                                llm_test_case,
                                _show_indicator=show_metric_indicator,
                                _in_component=True,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(
                                    llm_test_case, _in_component=True
                                )
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise
                        metric_data = create_metric_data(metric)
                        api_span.metrics_data.append(metric_data)
                        api_test_case.update_status(metric_data.success)
                        update_pbar(progress, pbar_eval_id)

                trace_level_metrics_count = (
                    len(current_trace.metrics) if current_trace.metrics else 0
                )
                pbar_eval_id = add_pbar(
                    progress,
                    f"     🎯 Evaluating component(s) (#{count})",
                    total=count_metrics_in_trace(trace=current_trace)
                    + trace_level_metrics_count,
                )

                start_time = time.perf_counter()

                # Handle trace-level metrics
                if current_trace.metrics:
                    has_task_completion = any(
                        isinstance(metric, TaskCompletionMetric)
                        for metric in current_trace.metrics
                    )

                    llm_test_case = None
                    if current_trace.input:
                        llm_test_case = LLMTestCase(
                            input=str(current_trace.input),
                            actual_output=(
                                str(current_trace.output)
                                if current_trace.output is not None
                                else None
                            ),
                            expected_output=current_trace.expected_output,
                            context=current_trace.context,
                            retrieval_context=current_trace.retrieval_context,
                            tools_called=current_trace.tools_called,
                            expected_tools=current_trace.expected_tools,
                        )
                    if llm_test_case is None and not has_task_completion:
                        raise ValueError(
                            "Unable to run metrics on trace without LLMTestCase. Are you sure you called `update_current_trace()`?"
                        )

                    if has_task_completion:
                        if llm_test_case is None:
                            llm_test_case = LLMTestCase(
                                input="None", actual_output="None"
                            )
                        llm_test_case._trace_dict = (
                            trace_manager.create_nested_spans_dict(
                                current_trace.root_spans[0]
                            )
                        )

                    for metric in current_trace.metrics:
                        metric.skipped = False
                        metric.error = None
                        if verbose_mode is not None:
                            metric.verbose_mode = verbose_mode

                    trace_api.metrics_data = []
                    for metric in current_trace.metrics:
                        try:
                            metric.measure(
                                llm_test_case,
                                _show_indicator=show_metric_indicator,
                                _in_component=True,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(
                                    llm_test_case, _in_component=True
                                )
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise

                        if not metric.skipped:
                            metric_data = create_metric_data(metric)
                            trace_api.metrics_data.append(metric_data)
                            api_test_case.update_metric_data(metric_data)
                            api_test_case.update_status(metric_data.success)
                            update_pbar(progress, pbar_eval_id)

                # Then handle span-level metrics
                dfs(current_trace.root_spans[0], progress, pbar_eval_id)
                end_time = time.perf_counter()
                run_duration = end_time - start_time

                # Update test run
                api_test_case.update_run_duration(run_duration)
                test_run_manager.update_test_run(api_test_case, test_case)
                test_results.append(create_test_result(api_test_case))

                update_pbar(progress, pbar_id)

    if show_indicator and _use_bar_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        with progress:
            pbar_id = add_pbar(
                progress,
                f"Running Component-Level Evals (sync)",
                total=len(goldens) * 2,
            )
            yield from evaluate_test_cases(progress=progress, pbar_id=pbar_id)
    else:
        yield from evaluate_test_cases()

    local_trace_manager.evaluating = False


def a_execute_agentic_test_cases_from_loop(
    goldens: List[Golden],
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    test_results: List[TestResult],
    loop: asyncio.AbstractEventLoop,
    throttle_value: int,
    max_concurrent: int,
    save_to_disk: bool = False,
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> Iterator[TestResult]:

    semaphore = asyncio.Semaphore(max_concurrent)
    original_create_task = asyncio.create_task

    test_run_manager = global_test_run_manager
    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run(identifier=identifier)

    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True
    local_trace_manager.evaluation_loop = True

    async def execute_callback_with_semaphore(coroutine: Awaitable):
        async with semaphore:
            return await coroutine

    def evaluate_test_cases(
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
        pbar_callback_id: Optional[int] = None,
    ):
        def create_callback_task(coro, **kwargs):
            task = loop.create_task(execute_callback_with_semaphore(coro))

            def on_task_done(t: asyncio.Task):
                update_pbar(progress, pbar_callback_id)
                update_pbar(progress, pbar_id)

            task.add_done_callback(on_task_done)
            return task

        asyncio.create_task = create_callback_task

        try:
            for golden in goldens:
                yield golden
                if global_evaluation_tasks.num_tasks() == 0:
                    update_pbar(progress, pbar_callback_id)
                    update_pbar(progress, pbar_id)
        finally:
            asyncio.create_task = original_create_task

        if global_evaluation_tasks.num_tasks() > 0:
            loop.run_until_complete(
                asyncio.gather(
                    *global_evaluation_tasks.get_tasks(),
                )
            )

        # Evaluate traces
        asyncio.create_task = loop.create_task
        if trace_manager.traces_to_evaluate:
            loop.run_until_complete(
                evaluate_traces(
                    traces_to_evaluate=trace_manager.traces_to_evaluate,
                    goldens=goldens,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    verbose_mode=verbose_mode,
                    ignore_errors=ignore_errors,
                    skip_on_missing_params=skip_on_missing_params,
                    show_indicator=show_indicator,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                    progress=progress,
                    pbar_id=pbar_id,
                    throttle_value=throttle_value,
                    max_concurrent=max_concurrent,
                )
            )
        elif openai_test_case_pairs:
            loop.run_until_complete(
                evaluate_test_case_pairs(
                    test_case_pairs=openai_test_case_pairs,
                    test_run=test_run,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    ignore_errors=ignore_errors,
                    skip_on_missing_params=skip_on_missing_params,
                    show_indicator=show_indicator,
                    verbose_mode=verbose_mode,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                    progress=progress,
                    pbar_id=pbar_id,
                    throttle_value=throttle_value,
                    max_concurrent=max_concurrent,
                )
            )
        elif trace_manager.integration_traces_to_evaluate:
            loop.run_until_complete(
                evaluate_traces(
                    traces_to_evaluate=trace_manager.integration_traces_to_evaluate,
                    goldens=goldens,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    verbose_mode=verbose_mode,
                    ignore_errors=ignore_errors,
                    skip_on_missing_params=skip_on_missing_params,
                    show_indicator=show_indicator,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                    progress=progress,
                    pbar_id=pbar_id,
                    throttle_value=throttle_value,
                    max_concurrent=max_concurrent,
                )
            )
        elif trace_manager.test_case_metrics:
            loop.run_until_complete(
                evaluate_test_case_pairs(
                    test_case_pairs=trace_manager.test_case_metrics,
                    test_run=test_run,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    ignore_errors=ignore_errors,
                    skip_on_missing_params=skip_on_missing_params,
                    show_indicator=show_indicator,
                    verbose_mode=verbose_mode,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                    progress=progress,
                    pbar_id=pbar_id,
                    throttle_value=throttle_value,
                    max_concurrent=max_concurrent,
                )
            )

    if show_indicator and _use_bar_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        with progress:
            pbar_id = add_pbar(
                progress,
                f"Running Component-Level Evals (sync)",
                total=len(goldens) * 2,
            )
            pbar_callback_id = add_pbar(
                progress,
                f"\t⚡ Calling LLM app (with {len(goldens)} goldens)",
                total=len(goldens),
            )
            yield from evaluate_test_cases(
                progress=progress,
                pbar_id=pbar_id,
                pbar_callback_id=pbar_callback_id,
            )
    else:
        yield from evaluate_test_cases()

    # Clean up
    local_trace_manager.evaluating = False
    local_trace_manager.traces_to_evaluate_order = []
    local_trace_manager.traces_to_evaluate = []
    global_evaluation_tasks.clear_tasks()
    asyncio.create_task = original_create_task


async def evaluate_traces(
    traces_to_evaluate: List[Trace],
    goldens: List[Golden],
    test_run_manager: TestRunManager,
    test_results: List[TestResult],
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    progress: Optional[Progress],
    pbar_id: Optional[int],
    throttle_value: int,
    max_concurrent: int,
):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_evals_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    eval_tasks = []
    for count, trace in enumerate(traces_to_evaluate):
        golden = goldens[count]
        with capture_evaluation_run("golden"):
            task = execute_evals_with_semaphore(
                func=a_execute_agentic_test_case,
                golden=golden,
                trace=trace,
                test_run_manager=test_run_manager,
                test_results=test_results,
                count=count,
                verbose_mode=verbose_mode,
                ignore_errors=ignore_errors,
                skip_on_missing_params=skip_on_missing_params,
                show_indicator=show_indicator,
                _use_bar_indicator=_use_bar_indicator,
                _is_assert_test=_is_assert_test,
                progress=progress,
                pbar_id=pbar_id,
            )
            eval_tasks.append(asyncio.create_task(task))
            await asyncio.sleep(throttle_value)
    await asyncio.gather(*eval_tasks)


async def evaluate_test_case_pairs(
    test_case_pairs: List[TestCaseMetricPair],
    test_run: TestRun,
    test_run_manager: TestRunManager,
    test_results: List[TestResult],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    verbose_mode: Optional[bool],
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    progress: Optional[Progress],
    pbar_id: Optional[int],
    throttle_value: int,
    max_concurrent: int,
):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    tasks = []
    for count, test_case_pair in enumerate(test_case_pairs):
        with capture_evaluation_run("test case"):
            if len(test_case_pair.metrics) == 0:
                update_pbar(progress, pbar_id)
                continue
            if verbose_mode is not None:
                for metric in test_case_pair.metrics:
                    metric.verbose_mode = verbose_mode
            copied_llm_metrics: List[BaseMetric] = copy_metrics(
                test_case_pair.metrics
            )
            task = execute_with_semaphore(
                func=a_execute_llm_test_cases,
                metrics=copied_llm_metrics,
                test_case=test_case_pair.test_case,
                test_run_manager=test_run_manager,
                test_results=test_results,
                count=count,
                test_run=test_run,
                ignore_errors=ignore_errors,
                skip_on_missing_params=skip_on_missing_params,
                use_cache=False,
                show_indicator=show_indicator,
                _use_bar_indicator=_use_bar_indicator,
                _is_assert_test=_is_assert_test,
                progress=progress,
                pbar_id=pbar_id,
            )
            tasks.append(asyncio.create_task(task))
            await asyncio.sleep(throttle_value)
    await asyncio.gather(*tasks)
