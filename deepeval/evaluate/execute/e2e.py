import logging

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
)
from copy import deepcopy
import asyncio
import time

from deepeval.evaluate.configs import (
    ErrorConfig,
    DisplayConfig,
    CacheConfig,
    AsyncConfig,
)
from deepeval.metrics.utils import copy_metrics
from deepeval.utils import (
    get_per_task_timeout_seconds,
    get_gather_timeout,
)
from deepeval.telemetry import capture_evaluation_run
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
)
from deepeval.metrics.indicator import (
    measure_metrics_with_indicator,
)
from deepeval.models.retry_policy import (
    set_outer_deadline,
    reset_outer_deadline,
    run_sync_with_timeout,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
)
from deepeval.test_case.api import create_api_test_case
from deepeval.test_run import (
    global_test_run_manager,
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
)
from deepeval.utils import add_pbar, update_pbar, custom_console
from deepeval.tracing.types import TestCaseMetricPair
from deepeval.config.settings import get_settings

logger = logging.getLogger(__name__)


from deepeval.evaluate.execute._common import (
    _await_with_outer_deadline,
    _execute_metric,
    _log_gather_timeout,
    _timeout_msg,
)


def execute_test_cases(
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
    metrics: Union[
        List[BaseMetric],
        List[BaseConversationalMetric],
    ],
    error_config: Optional[ErrorConfig] = ErrorConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    identifier: Optional[str] = None,
    test_run_manager: Optional[TestRunManager] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:
    global_test_run_cache_manager.disable_write_cache = (
        cache_config.write_cache is False
    )

    if test_run_manager is None:
        test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = cache_config.write_cache
    test_run = test_run_manager.get_test_run(identifier=identifier)
    if test_run is None:
        # ensure we have a test_run ( in case it couldn't be loaded from disk )
        test_run_manager.create_test_run(identifier=identifier)
        test_run = test_run_manager.get_test_run(identifier=identifier)

    # capture once for inner closures
    hyperparameters = test_run.hyperparameters if test_run is not None else None

    if display_config.verbose_mode is not None:
        for metric in metrics:
            metric.verbose_mode = display_config.verbose_mode

    conversational_metrics: List[BaseConversationalMetric] = []
    llm_metrics: List[BaseMetric] = []
    for metric in metrics:
        metric.async_mode = False
        if isinstance(metric, BaseMetric):
            llm_metrics.append(metric)
        elif isinstance(metric, BaseConversationalMetric):
            conversational_metrics.append(metric)

    test_results: List[TestResult] = []

    def evaluate_test_cases(
        progress: Optional[Progress] = None, pbar_id: Optional[int] = None
    ):
        llm_test_case_count = -1
        conversational_test_case_count = -1
        show_metric_indicator = (
            display_config.show_indicator and not _use_bar_indicator
        )
        for i, test_case in enumerate(test_cases):
            # skip what we know we won't run
            if isinstance(test_case, LLMTestCase):
                if not llm_metrics:
                    update_pbar(progress, pbar_id)
                    continue
                per_case_total = len(llm_metrics)
            elif isinstance(test_case, ConversationalTestCase):
                if not conversational_metrics:
                    update_pbar(progress, pbar_id)
                    continue
                per_case_total = len(conversational_metrics)

            pbar_test_case_id = add_pbar(
                progress,
                f"    🎯 Evaluating test case #{i}",
                total=per_case_total,
            )

            metrics_for_case = (
                llm_metrics
                if (isinstance(test_case, LLMTestCase))
                else conversational_metrics
            )
            api_test_case = create_api_test_case(
                test_case=test_case,
                index=(
                    llm_test_case_count + 1
                    if (isinstance(test_case, LLMTestCase))
                    else (conversational_test_case_count + 1)
                ),
            )
            emitted = [False] * len(metrics_for_case)
            index_of = {id(m): i for i, m in enumerate(metrics_for_case)}
            current_index = -1
            start_time = time.perf_counter()
            deadline_timeout = get_per_task_timeout_seconds()
            deadline_token = set_outer_deadline(deadline_timeout)
            new_cached_test_case: CachedTestCase = None
            try:

                def _run_case():
                    nonlocal new_cached_test_case, current_index, llm_test_case_count, conversational_test_case_count
                    with capture_evaluation_run("test case"):
                        for metric in metrics:
                            metric.error = None  # Reset metric error

                        if isinstance(test_case, LLMTestCase):
                            llm_test_case_count += 1
                            cached_test_case = None
                            if cache_config.use_cache:
                                cached_test_case = global_test_run_cache_manager.get_cached_test_case(
                                    test_case, hyperparameters
                                )

                            ##### Metric Calculation #####
                            new_cached_test_case = CachedTestCase()

                            for metric in llm_metrics:
                                current_index = index_of[id(metric)]
                                metric_data = None
                                if cached_test_case is not None:
                                    cached_metric_data = Cache.get_metric_data(
                                        metric, cached_test_case
                                    )
                                    if cached_metric_data:
                                        metric_data = (
                                            cached_metric_data.metric_data
                                        )

                                if metric_data is None:
                                    res = _execute_metric(
                                        metric=metric,
                                        test_case=test_case,
                                        show_metric_indicator=show_metric_indicator,
                                        in_component=False,
                                        error_config=error_config,
                                    )
                                    if res == "skip":
                                        continue
                                    metric_data = create_metric_data(metric)

                                # here, we will check for an additional property on the flattened test cases to see if updating is necessary
                                api_test_case.update_metric_data(metric_data)
                                emitted[current_index] = True
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

                        # No caching for conversational metrics yet
                        elif isinstance(test_case, ConversationalTestCase):
                            conversational_test_case_count += 1
                            for metric in conversational_metrics:
                                current_index = index_of[id(metric)]
                                res = _execute_metric(
                                    metric=metric,
                                    test_case=test_case,
                                    show_metric_indicator=show_metric_indicator,
                                    in_component=False,
                                    error_config=error_config,
                                )
                                if res == "skip":
                                    continue

                                metric_data = create_metric_data(metric)
                                api_test_case.update_metric_data(metric_data)
                                emitted[current_index] = True
                                update_pbar(progress, pbar_test_case_id)

                run_sync_with_timeout(_run_case, deadline_timeout)
            except (asyncio.TimeoutError, TimeoutError):

                msg = _timeout_msg("evaluating metric", deadline_timeout)
                for i, metric in enumerate(metrics_for_case):
                    if metric.skipped:
                        continue
                    # already finished or errored? leave it
                    if metric.success is not None or metric.error is not None:
                        continue
                    if i == current_index:
                        metric.success = False
                        metric.error = msg
                    elif i > current_index:
                        metric.success = False
                        metric.error = "Skipped due to case timeout."

                if not error_config.ignore_errors:
                    raise

            finally:
                try:
                    if (
                        isinstance(test_case, LLMTestCase)
                        and new_cached_test_case is not None
                    ):
                        ### Cache Test Run ###
                        global_test_run_cache_manager.cache_test_case(
                            test_case,
                            new_cached_test_case,
                            hyperparameters,
                        )
                        global_test_run_cache_manager.cache_test_case(
                            test_case,
                            new_cached_test_case,
                            hyperparameters,
                            to_temp=True,
                        )

                    # Attach MetricData for *all* metrics (finished or synthesized)
                    for i, metric in enumerate(metrics_for_case):
                        if metric.skipped:
                            continue
                        if not emitted[i]:
                            api_test_case.update_metric_data(
                                create_metric_data(metric)
                            )

                    elapsed = time.perf_counter() - start_time
                    api_test_case.update_run_duration(
                        elapsed if elapsed >= 0 else deadline_timeout
                    )
                    test_run_manager.update_test_run(api_test_case, test_case)
                    test_results.append(create_test_result(api_test_case))
                    update_pbar(progress, pbar_id)
                finally:
                    reset_outer_deadline(deadline_token)

    if display_config.show_indicator and _use_bar_indicator:
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
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
    metrics: Union[
        List[BaseMetric],
        List[BaseConversationalMetric],
    ],
    error_config: Optional[ErrorConfig] = ErrorConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    identifier: Optional[str] = None,
    test_run_manager: Optional[TestRunManager] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:
    semaphore = asyncio.Semaphore(async_config.max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await _await_with_outer_deadline(
                func, *args, timeout=get_per_task_timeout_seconds(), **kwargs
            )

    global_test_run_cache_manager.disable_write_cache = (
        cache_config.write_cache is False
    )
    if test_run_manager is None:
        test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = cache_config.write_cache
    test_run = test_run_manager.get_test_run(identifier=identifier)

    if display_config.verbose_mode is not None:
        for metric in metrics:
            metric.verbose_mode = display_config.verbose_mode

    llm_metrics: List[BaseMetric] = []
    conversational_metrics: List[BaseConversationalMetric] = []
    for metric in metrics:
        if isinstance(metric, BaseMetric):
            llm_metrics.append(metric)
        elif isinstance(metric, BaseConversationalMetric):
            conversational_metrics.append(metric)

    llm_test_case_counter = -1
    conversational_test_case_counter = -1
    test_results: List[Union[TestResult, LLMTestCase]] = []
    tasks = []

    if display_config.show_indicator and _use_bar_indicator:
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
                            func=_a_execute_llm_test_cases,
                            metrics=copied_llm_metrics,
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=llm_test_case_counter,
                            test_run=test_run,
                            ignore_errors=error_config.ignore_errors,
                            skip_on_missing_params=error_config.skip_on_missing_params,
                            use_cache=cache_config.use_cache,
                            show_indicator=display_config.show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            _is_assert_test=_is_assert_test,
                            progress=progress,
                            pbar_id=pbar_id,
                        )
                        tasks.append(asyncio.create_task(task))

                    elif isinstance(test_case, ConversationalTestCase):
                        conversational_test_case_counter += 1

                        task = execute_with_semaphore(
                            func=_a_execute_conversational_test_cases,
                            metrics=copy_metrics(conversational_metrics),
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=conversational_test_case_counter,
                            ignore_errors=error_config.ignore_errors,
                            skip_on_missing_params=error_config.skip_on_missing_params,
                            show_indicator=display_config.show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            _is_assert_test=_is_assert_test,
                            progress=progress,
                            pbar_id=pbar_id,
                        )
                        tasks.append(asyncio.create_task(task))

                    await asyncio.sleep(async_config.throttle_value)

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=get_gather_timeout(),
                )
            except (asyncio.TimeoutError, TimeoutError) as e:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

                _log_gather_timeout(logger, exc=e)

                if not error_config.ignore_errors:
                    raise

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
                        func=_a_execute_llm_test_cases,
                        metrics=copied_llm_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=llm_test_case_counter,
                        test_run=test_run,
                        ignore_errors=error_config.ignore_errors,
                        skip_on_missing_params=error_config.skip_on_missing_params,
                        use_cache=cache_config.use_cache,
                        _use_bar_indicator=_use_bar_indicator,
                        _is_assert_test=_is_assert_test,
                        show_indicator=display_config.show_indicator,
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
                        func=_a_execute_conversational_test_cases,
                        metrics=copied_conversational_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=conversational_test_case_counter,
                        ignore_errors=error_config.ignore_errors,
                        skip_on_missing_params=error_config.skip_on_missing_params,
                        _use_bar_indicator=_use_bar_indicator,
                        _is_assert_test=_is_assert_test,
                        show_indicator=display_config.show_indicator,
                    )
                    tasks.append(asyncio.create_task((task)))

                await asyncio.sleep(async_config.throttle_value)

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=get_gather_timeout(),
            )
        except (asyncio.TimeoutError, TimeoutError):
            # Cancel any still-pending tasks and drain them
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not error_config.ignore_errors:
                raise

    return test_results


async def _a_execute_llm_test_cases(
    metrics: List[BaseMetric],
    test_case: LLMTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, LLMTestCase]],
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
    logger.info("in _a_execute_llm_test_cases")
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
    try:
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
    except asyncio.CancelledError:
        if get_settings().DEEPEVAL_DISABLE_TIMEOUTS:
            msg = (
                "Cancelled while evaluating metric. "
                "(DeepEval timeouts are disabled; this cancellation likely came from upstream orchestration or manual cancellation). "
                "Set DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
            )
        else:
            msg = (
                "Timed out/cancelled while evaluating metric. "
                "Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set "
                "DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
            )
        for m in metrics:
            if getattr(m, "skipped", False):
                continue
            # If the task never finished and didn't set a terminal state, mark it now
            if getattr(m, "success", None) is None and not getattr(
                m, "error", None
            ):
                m.success = False
                m.error = msg
        if not ignore_errors:
            raise
    finally:
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
                    metric_configuration=Cache.create_metric_configuration(
                        metric
                    ),
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


async def _a_execute_conversational_test_cases(
    metrics: List[Union[BaseMetric, BaseConversationalMetric]],
    test_case: ConversationalTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, LLMTestCase]],
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

    try:
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

    except asyncio.CancelledError:
        if get_settings().DEEPEVAL_DISABLE_TIMEOUTS:
            msg = (
                "Cancelled while evaluating metric. "
                "(DeepEval timeouts are disabled; this cancellation likely came from upstream orchestration or manual cancellation). "
                "Set DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
            )
        else:
            msg = (
                "Timed out/cancelled while evaluating metric. "
                "Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set "
                "DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
            )
        for m in metrics:
            if getattr(m, "skipped", False):
                continue
            # If the task never finished and didn't set a terminal state, mark it now
            if getattr(m, "success", None) is None and not getattr(
                m, "error", None
            ):
                m.success = False
                m.error = msg
        if not ignore_errors:
            raise

    finally:
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


async def _evaluate_test_case_pairs(
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
            return await _await_with_outer_deadline(
                func, *args, timeout=get_per_task_timeout_seconds(), **kwargs
            )

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
                func=_a_execute_llm_test_cases,
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

    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=get_gather_timeout(),
        )
    except (asyncio.TimeoutError, TimeoutError):
        # Cancel any still-pending tasks and drain them
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
