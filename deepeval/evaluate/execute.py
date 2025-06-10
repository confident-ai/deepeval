from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from typing import Callable, List, Optional, Union, Any, Awaitable
from rich.console import Console
from rich.theme import Theme
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
)
from deepeval.utils import add_pbar, update_pbar, custom_console


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
                    actual_output=golden.actual_output or "TODO",
                    expected_output=golden.expected_output,
                    context=golden.context,
                    retrieval_context=golden.retrieval_context,
                    additional_metadata=golden.additional_metadata,
                    tools_called=golden.tools_called,
                    expected_tools=golden.expected_tools,
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
                    test_case: LLMTestCase = span.llm_test_case
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

                    if span.metrics == None or span.llm_test_case == None:
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
                                test_case,
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
                                metric.measure(test_case, _in_component=True)
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

                pbar_eval_id = add_pbar(
                    progress,
                    f"     🎯 Evaluating component(s) (#{count})",
                    total=count_metrics_in_trace(trace=current_trace),
                )

                start_time = time.perf_counter()
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
    observed_callback: Union[
        Callable[[str], Any], Callable[[str], Awaitable[Any]]
    ],
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
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

    pbar_eval_id = add_pbar(
        progress,
        f"     🎯 Evaluating component(s) (#{count})",
        total=count_metrics_in_trace(trace=current_trace),
    )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=golden.actual_output,
        expected_output=golden.expected_output,
        context=golden.context,
        retrieval_context=golden.retrieval_context,
        additional_metadata=golden.additional_metadata,
        tools_called=golden.tools_called,
        expected_tools=golden.expected_tools,
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
    if span.llm_test_case is None:
        raise ValueError(
            "Unable to run metrics on span without LLMTestCase. Are you sure you called `update_current_span()`?"
        )

    show_metrics_indicator = show_indicator and not _use_bar_indicator
    metrics: List[BaseMetric] = span.metrics
    test_case: LLMTestCase = span.llm_test_case

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
