import inspect
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
    Any,
    Awaitable,
    Iterator,
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
from deepeval.tracing.tracing import (
    Observer,
    trace_manager,
    Trace,
    BaseSpan,
    AgentSpan,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
)
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.api import (
    TraceApi,
    BaseApiSpan,
)
from deepeval.dataset import Golden
from deepeval.contextvars import set_current_golden, reset_current_golden
from deepeval.errors import MissingTestCaseParamsError, DeepEvalError
from deepeval.metrics.utils import copy_metrics
from deepeval.utils import (
    get_or_create_event_loop,
    shorten,
    len_medium,
    format_error_text,
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
from deepeval.models.retry_policy import (
    set_outer_deadline,
    reset_outer_deadline,
    run_sync_with_timeout,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
)
from deepeval.test_case.api import create_api_test_case
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
    count_observe_decorators_in_module,
    create_api_trace,
    create_metric_data,
    create_test_result,
    count_metrics_in_trace,
    count_total_metrics_for_trace,
    count_metrics_in_span_subtree,
    extract_trace_test_results,
)
from deepeval.utils import add_pbar, update_pbar, custom_console
from deepeval.tracing.types import TestCaseMetricPair, TraceSpanStatus
from deepeval.tracing.api import TraceSpanApiStatus
from deepeval.config.settings import get_settings
from deepeval.test_run import TEMP_FILE_PATH
from deepeval.confident.api import is_confident
from deepeval.test_run.hyperparameters import (
    process_hyperparameters,
    process_prompts,
)

logger = logging.getLogger(__name__)


def _skip_metrics_for_error(
    span: Optional[BaseSpan] = None,
    trace: Optional[Trace] = None,
) -> bool:
    # trace failure: skip everything under this trace
    if trace is not None and trace.status == TraceSpanStatus.ERRORED:
        return True
    # span failure: skip this span’s metrics
    if span is not None and span.status == TraceSpanStatus.ERRORED:
        return True
    return False


def _trace_error(current_trace: Trace) -> Optional[str]:
    def _first_err(s: BaseSpan) -> Optional[str]:
        if s.status == TraceSpanStatus.ERRORED and s.error:
            return s.error
        for c in s.children or []:
            e = _first_err(c)
            if e:
                return e
        return None

    for root in current_trace.root_spans or []:
        e = _first_err(root)
        if e:
            return e
    return None


def _get_trace_by_uuid_anywhere(trace_uuid: str):
    """
    Resolver for a trace UUID across the manager's state.

    First tries the manager's indexed lookup, which (covers active/in-flight traces,
    then does a linear scan of the full `trace_manager.traces` list, which covers
    traces that were recorded/closed earlier or not yet indexed. Returns
    the concrete Trace object or None if not found.
    """
    tr = trace_manager.get_trace_by_uuid(trace_uuid)
    if tr:
        return tr
    for tr in trace_manager.traces:
        if tr.uuid == trace_uuid:
            return tr
    return None


def _pick_root_for_marking(trace):
    """
    Choose the most appropriate root span to annotate on error/cancel.

    Heuristic:
      - Prefer the most recent open root, which will have no `end_time` since this is the
        span currently in flight.
      - If none are open, use the last root span if it exists.
      - If the trace has no roots, return None.

    This favors marking the active root in multi root traces while remaining
    stable for already closed traces.
    """
    open_roots = [rs for rs in trace.root_spans if rs.end_time is None]
    return (
        open_roots[-1]
        if open_roots
        else (trace.root_spans[-1] if trace.root_spans else None)
    )


def _resolve_trace_and_root_for_task(t: asyncio.Task):
    """
    Resolve trace and root for a completed task using the weak binding map.

    Steps:
      1. Look up the task in `trace_manager.task_bindings` to get the
         bound `trace_uuid` and, if available, `root_span_uuid`.
      2. Resolve the Trace with `_get_trace_by_uuid_anywhere`.
      3. If a bound root UUID exists, try to find that exact root on the trace.
      4. Otherwise, fall back to `_pick_root_for_marking(trace)`.

    Returns a trace / root tuple. Either may be `None` when no binding is
    present. This function is used by `on_task_done` to robustly mark error/cancel
    states without assuming a single root trace or a root that is still open.
    """
    binding = trace_manager.task_bindings.get(t) or {}
    trace_uuid = binding.get("trace_uuid")
    root_span_uuid = binding.get("root_span_uuid")

    trace = _get_trace_by_uuid_anywhere(trace_uuid) if trace_uuid else None
    root = None

    if trace and root_span_uuid:
        root = next(
            (rs for rs in trace.root_spans if rs.uuid == root_span_uuid), None
        )

    if trace and root is None:
        root = _pick_root_for_marking(trace)

    return trace, root


async def _snapshot_tasks():
    cur = asyncio.current_task()
    # `all_tasks` returns tasks for the current running loop only
    return {t for t in asyncio.all_tasks() if t is not cur}


def _per_task_timeout() -> float:
    return get_settings().DEEPEVAL_PER_TASK_TIMEOUT_SECONDS


def _gather_timeout() -> float:
    s = get_settings()
    return (
        s.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS
        + s.DEEPEVAL_TASK_GATHER_BUFFER_SECONDS
    )


def filter_duplicate_results(
    main_result: TestResult, results: List[TestResult]
) -> List[TestResult]:
    return [
        result
        for result in results
        if not (
            (result.input == main_result.input)
            and (result.actual_output == main_result.actual_output)
            and (result.metrics_data == main_result.metrics_data)
        )
    ]


async def _await_with_outer_deadline(obj, *args, timeout: float, **kwargs):
    token = set_outer_deadline(timeout)
    try:
        if inspect.isawaitable(obj):
            coro = obj
        else:
            coro = obj(*args, **kwargs)
        return await asyncio.wait_for(coro, timeout=timeout)
    finally:
        reset_outer_deadline(token)


###########################################
### E2E Evals #############################
###########################################


def execute_test_cases(
    test_cases: Union[
        List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
    ],
    metrics: Union[
        List[BaseMetric],
        List[BaseConversationalMetric],
        List[BaseMultimodalMetric],
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
        progress: Optional[Progress] = None, pbar_id: Optional[int] = None
    ):
        llm_test_case_count = -1
        mllm_test_case_count = -1
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
            elif isinstance(test_case, MLLMTestCase):
                if not mllm_metrics:
                    update_pbar(progress, pbar_id)
                    continue
                per_case_total = len(mllm_metrics)
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
                if isinstance(test_case, LLMTestCase)
                else (
                    mllm_metrics
                    if isinstance(test_case, MLLMTestCase)
                    else conversational_metrics
                )
            )
            api_test_case = create_api_test_case(
                test_case=test_case,
                index=(
                    llm_test_case_count + 1
                    if isinstance(test_case, LLMTestCase)
                    else (
                        mllm_test_case_count + 1
                        if isinstance(test_case, MLLMTestCase)
                        else conversational_test_case_count + 1
                    )
                ),
            )
            emitted = [False] * len(metrics_for_case)
            index_of = {id(m): i for i, m in enumerate(metrics_for_case)}
            current_index = -1
            start_time = time.perf_counter()
            deadline_timeout = _per_task_timeout()
            deadline_token = set_outer_deadline(deadline_timeout)
            new_cached_test_case: CachedTestCase = None
            try:

                def _run_case():
                    nonlocal new_cached_test_case, current_index, llm_test_case_count, mllm_test_case_count, conversational_test_case_count
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

                        # No caching and not sending test cases to Confident AI for multimodal metrics yet
                        elif isinstance(test_case, MLLMTestCase):
                            mllm_test_case_count += 1
                            for metric in mllm_metrics:
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
                msg = (
                    f"Timed out after {deadline_timeout:.2f}s while evaluating metric. "
                    "Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set "
                    "DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
                )
                for i, m in enumerate(metrics_for_case):
                    if getattr(m, "skipped", False):
                        continue
                    # already finished or errored? leave it
                    if getattr(m, "success", None) is not None or getattr(
                        m, "error", None
                    ):
                        continue
                    if i == current_index:
                        m.success = False
                        m.error = msg
                    elif i > current_index:
                        m.success = False
                        m.error = "Skipped due to case timeout."

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
                    for i, m in enumerate(metrics_for_case):
                        if getattr(m, "skipped", False):
                            continue
                        if not emitted[i]:
                            api_test_case.update_metric_data(
                                create_metric_data(m)
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
    test_cases: Union[
        List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
    ],
    metrics: Union[
        List[BaseMetric],
        List[BaseConversationalMetric],
        List[BaseMultimodalMetric],
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
            timeout = _per_task_timeout()
            return await _await_with_outer_deadline(
                func, *args, timeout=timeout, **kwargs
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

                    elif isinstance(test_case, MLLMTestCase):
                        mllm_test_case_counter += 1
                        copied_multimodal_metrics: List[
                            BaseMultimodalMetric
                        ] = copy_metrics(mllm_metrics)
                        task = execute_with_semaphore(
                            func=_a_execute_mllm_test_cases,
                            metrics=copied_multimodal_metrics,
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=mllm_test_case_counter,
                            ignore_errors=error_config.ignore_errors,
                            skip_on_missing_params=error_config.skip_on_missing_params,
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
                    timeout=_gather_timeout(),
                )
            except (asyncio.TimeoutError, TimeoutError):
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                logging.getLogger("deepeval").error(
                    "Gather timed out after %.1fs. Some metrics may be marked as timed out.",
                    _gather_timeout(),
                )

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

                elif isinstance(test_case, MLLMTestCase):
                    mllm_test_case_counter += 1
                    copied_multimodal_metrics: List[BaseMultimodalMetric] = (
                        copy_metrics(mllm_metrics)
                    )
                    task = execute_with_semaphore(
                        func=_a_execute_mllm_test_cases,
                        metrics=copied_multimodal_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=mllm_test_case_counter,
                        ignore_errors=error_config.ignore_errors,
                        skip_on_missing_params=error_config.skip_on_missing_params,
                        _use_bar_indicator=_use_bar_indicator,
                        _is_assert_test=_is_assert_test,
                        show_indicator=display_config.show_indicator,
                    )
                    tasks.append(asyncio.create_task(task))

                await asyncio.sleep(async_config.throttle_value)

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=_gather_timeout(),
            )
        except (asyncio.TimeoutError, TimeoutError):
            # Cancel any still-pending tasks and drain them
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    return test_results


async def _a_execute_llm_test_cases(
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


async def _a_execute_mllm_test_cases(
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
        run_duration = test_end_time - test_start_time
        api_test_case.update_run_duration(run_duration)

        ### Update Test Run ###
        test_run_manager.update_test_run(api_test_case, test_case)
        test_results.append(create_test_result(api_test_case))
        update_pbar(progress, pbar_id)


async def _a_execute_conversational_test_cases(
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


###########################################
### Component-Level Evals #################
###########################################


def execute_agentic_test_cases(
    goldens: List[Golden],
    observed_callback: Union[
        Callable[[str], Any], Callable[[str], Awaitable[Any]]
    ],
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:

    test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = cache_config.write_cache
    test_run = test_run_manager.get_test_run(identifier=identifier)
    if test_run is None:
        # Create if not found
        test_run_manager.create_test_run(identifier=identifier)
        test_run = test_run_manager.get_test_run(identifier=identifier)

    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True
    test_results: List[TestResult] = []

    def evaluate_test_cases(
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ):
        count = -1
        show_metric_indicator = (
            display_config.show_indicator and not _use_bar_indicator
        )

        for golden in goldens:
            count += 1

            pbar_case_increments = (
                0  # tracks how many times we advance `pbar_id` for this golden
            )
            emitted_trace = set()
            current_trace: Optional[Trace] = None
            trace_api = None
            api_test_case = None
            test_case = None

            def _run_golden():
                nonlocal current_trace, trace_api, api_test_case, test_case, pbar_case_increments
                # keep the evaluation context inside the timed function
                with capture_evaluation_run("golden"):
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
                            coro = observed_callback(golden.input)
                            loop.run_until_complete(
                                _await_with_outer_deadline(
                                    coro,
                                    timeout=_per_task_timeout(),
                                )
                            )
                        else:
                            observed_callback(golden.input)

                        # we have a trace now
                        current_trace = current_trace_context.get()

                    update_pbar(progress, pbar_tags_id, advance=total_tags)
                    update_pbar(progress, pbar_id)
                    pbar_case_increments += 1

                    # Create empty trace api for llm api test case
                    trace_api = create_api_trace(current_trace, golden)

                    # Build the test case and api test case
                    test_case = LLMTestCase(
                        input=golden.input,
                        actual_output=(
                            str(current_trace.output)
                            if current_trace
                            and current_trace.output is not None
                            else None
                        ),
                        expected_output=(
                            current_trace.expected_output
                            if current_trace
                            else None
                        ),
                        context=(
                            current_trace.context if current_trace else None
                        ),
                        retrieval_context=(
                            current_trace.retrieval_context
                            if current_trace
                            else None
                        ),
                        additional_metadata=golden.additional_metadata,
                        tools_called=(
                            current_trace.tools_called
                            if current_trace
                            else None
                        ),
                        expected_tools=(
                            current_trace.expected_tools
                            if current_trace
                            else None
                        ),
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

                    # DFS and trace metric evaluation
                    def dfs(
                        span: BaseSpan,
                        progress: Optional[Progress] = None,
                        pbar_eval_id: Optional[int] = None,
                    ):
                        metrics: List[BaseMetric] = list(span.metrics or [])
                        api_span: BaseApiSpan = (
                            trace_manager._convert_span_to_api_span(span)
                        )

                        if isinstance(span, AgentSpan):
                            trace_api.agent_spans.append(api_span)
                        elif isinstance(span, LlmSpan):
                            trace_api.llm_spans.append(api_span)
                            log_prompt(span, test_run_manager)
                        elif isinstance(span, RetrieverSpan):
                            trace_api.retriever_spans.append(api_span)
                        elif isinstance(span, ToolSpan):
                            trace_api.tool_spans.append(api_span)
                        else:
                            trace_api.base_spans.append(api_span)

                        if _skip_metrics_for_error(
                            span=span, trace=current_trace
                        ):
                            api_span.status = TraceSpanApiStatus.ERRORED
                            api_span.error = span.error or _trace_error(
                                current_trace
                            )
                            if progress and pbar_eval_id is not None:
                                update_pbar(
                                    progress,
                                    pbar_eval_id,
                                    advance=count_metrics_in_span_subtree(span),
                                )
                            return

                        # evaluate children first
                        for child in span.children:
                            dfs(child, progress, pbar_eval_id)

                        # If there are no metrics, then there is nothing to do on this span.
                        if not metrics:
                            return

                        has_task_completion = any(
                            isinstance(metric, TaskCompletionMetric)
                            for metric in metrics
                        )

                        requires_trace = any(
                            getattr(metric, "requires_trace", False)
                            for metric in metrics
                        )

                        llm_test_case = None
                        if span.input is not None:
                            llm_test_case = LLMTestCase(
                                input=str(span.input),
                                actual_output=(
                                    str(span.output)
                                    if span.output is not None
                                    else None
                                ),
                                expected_output=span.expected_output,
                                context=span.context,
                                retrieval_context=span.retrieval_context,
                                tools_called=span.tools_called,
                                expected_tools=span.expected_tools,
                            )

                        # If any metric needs a trace tree or a completion verdict, attach the trace
                        if has_task_completion or requires_trace:
                            if llm_test_case is None:
                                llm_test_case = LLMTestCase(input="None")
                            llm_test_case._trace_dict = (
                                trace_manager.create_nested_spans_dict(span)
                            )
                        else:
                            # Without a test case we cannot evaluate span metrics
                            if llm_test_case is None:
                                api_span.status = TraceSpanApiStatus.ERRORED
                                api_span.error = format_error_text(
                                    DeepEvalError(
                                        "Span has metrics but no LLMTestCase. "
                                        "Are you sure you called `update_current_span()`?"
                                    )
                                )
                                if progress and pbar_eval_id is not None:
                                    update_pbar(
                                        progress,
                                        pbar_eval_id,
                                        advance=count_metrics_in_span_subtree(
                                            span
                                        ),
                                    )
                                return

                        # Preparing metric calculation
                        api_span.metrics_data = []
                        for metric in metrics:
                            metric.skipped = False
                            metric.error = None
                            if display_config.verbose_mode is not None:
                                metric.verbose_mode = (
                                    display_config.verbose_mode
                                )

                        # Metric calculation
                        for metric in metrics:
                            res = _execute_metric(
                                metric=metric,
                                test_case=llm_test_case,
                                show_metric_indicator=show_metric_indicator,
                                in_component=True,
                                error_config=error_config,
                            )
                            if res == "skip":
                                continue
                            metric_data = create_metric_data(metric)
                            api_span.metrics_data.append(metric_data)
                            api_test_case.update_status(metric_data.success)
                            update_pbar(progress, pbar_eval_id)

                    trace_level_metrics_count = (
                        len(current_trace.metrics)
                        if current_trace and current_trace.metrics
                        else 0
                    )
                    pbar_eval_id = add_pbar(
                        progress,
                        f"     🎯 Evaluating component(s) (#{count})",
                        total=count_metrics_in_trace(trace=current_trace)
                        + trace_level_metrics_count,
                    )

                    start_time = time.perf_counter()

                    skip_metrics_for_this_golden = False
                    if _skip_metrics_for_error(trace=current_trace):
                        trace_api.status = TraceSpanApiStatus.ERRORED
                        if progress and pbar_eval_id is not None:
                            update_pbar(
                                progress,
                                pbar_eval_id,
                                advance=count_total_metrics_for_trace(
                                    current_trace
                                ),
                            )
                    else:
                        if current_trace and current_trace.metrics:
                            has_task_completion = any(
                                isinstance(metric, TaskCompletionMetric)
                                for metric in current_trace.metrics
                            )
                            requires_trace = any(
                                getattr(metric, "requires_trace", False)
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
                            if has_task_completion or requires_trace:
                                if llm_test_case is None:
                                    llm_test_case = LLMTestCase(input="None")
                                llm_test_case._trace_dict = (
                                    trace_manager.create_nested_spans_dict(
                                        current_trace.root_spans[0]
                                    )
                                )
                            else:
                                if llm_test_case is None:
                                    current_trace.status = (
                                        TraceSpanStatus.ERRORED
                                    )
                                    trace_api.status = (
                                        TraceSpanApiStatus.ERRORED
                                    )
                                    if current_trace.root_spans:
                                        current_trace.root_spans[0].status = (
                                            TraceSpanStatus.ERRORED
                                        )
                                        current_trace.root_spans[0].error = (
                                            format_error_text(
                                                DeepEvalError(
                                                    "Trace has metrics but no LLMTestCase (missing input/output). "
                                                    "Are you sure you called `update_current_trace()`?"
                                                )
                                            )
                                        )
                                    if progress and pbar_eval_id is not None:
                                        update_pbar(
                                            progress,
                                            pbar_eval_id,
                                            advance=count_total_metrics_for_trace(
                                                current_trace
                                            ),
                                        )
                                    skip_metrics_for_this_golden = True

                            if not skip_metrics_for_this_golden:
                                for metric in current_trace.metrics:
                                    metric.skipped = False
                                    metric.error = None
                                    if display_config.verbose_mode is not None:
                                        metric.verbose_mode = (
                                            display_config.verbose_mode
                                        )

                                trace_api.metrics_data = []
                                for metric in current_trace.metrics:
                                    res = _execute_metric(
                                        metric=metric,
                                        test_case=llm_test_case,
                                        show_metric_indicator=show_metric_indicator,
                                        in_component=True,
                                        error_config=error_config,
                                    )
                                    if res == "skip":
                                        continue

                                    if not metric.skipped:
                                        metric_data = create_metric_data(metric)
                                        trace_api.metrics_data.append(
                                            metric_data
                                        )
                                        api_test_case.update_metric_data(
                                            metric_data
                                        )
                                        api_test_case.update_status(
                                            metric_data.success
                                        )
                                        emitted_trace.add(id(metric))
                                        update_pbar(progress, pbar_eval_id)

                            # handle span metrics
                            dfs(
                                current_trace.root_spans[0],
                                progress,
                                pbar_eval_id,
                            )

                    # TODO: Do I need this block, or is it duplicated in finally?
                    end_time = time.perf_counter()
                    run_duration = end_time - start_time
                    api_test_case.update_run_duration(run_duration)
                    test_run_manager.update_test_run(api_test_case, test_case)
                    test_results.append(create_test_result(api_test_case))
                    test_results.extend(extract_trace_test_results(trace_api))
                    update_pbar(progress, pbar_id)
                    pbar_case_increments += 1

            # run the golden with a timeout
            start_time = time.perf_counter()
            deadline = _per_task_timeout()

            try:
                run_sync_with_timeout(_run_golden, deadline)
            except (asyncio.TimeoutError, TimeoutError):
                # mark any not yet finished trace level and span level metrics as timed out.
                msg = (
                    f"Timed out after {deadline:.2f}s while executing agentic test case. "
                    "Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set "
                    "DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
                )

                if current_trace is not None:
                    # Trace-level metrics
                    if getattr(current_trace, "metrics", None):
                        for m in current_trace.metrics:
                            if getattr(m, "skipped", False):
                                continue
                            # if already has a terminal state, leave it alone
                            if getattr(
                                m, "success", None
                            ) is not None or getattr(m, "error", None):
                                continue
                            m.success = False
                            m.error = msg

                    # span level metrics, walk the tree
                    def _walk(span):
                        for child in getattr(span, "children", []) or []:
                            _walk(child)
                        for m in list(getattr(span, "metrics", []) or []):
                            if getattr(m, "skipped", False):
                                continue
                            if getattr(
                                m, "success", None
                            ) is not None or getattr(m, "error", None):
                                continue
                            m.success = False
                            m.error = msg

                    for root in getattr(current_trace, "root_spans", []) or []:
                        _walk(root)

                # raise if we are not ignoring errors
                if not error_config.ignore_errors:
                    raise

            finally:
                try:
                    # Ensure we have an api_test_case to attach results to.
                    if api_test_case is None:
                        # build a minimal test_case
                        if test_case is None:
                            out = (
                                str(current_trace.output)
                                if (
                                    current_trace is not None
                                    and current_trace.output is not None
                                )
                                else None
                            )
                            test_case = LLMTestCase(
                                input=golden.input,
                                actual_output=out,
                                expected_output=(
                                    current_trace.expected_output
                                    if current_trace
                                    else None
                                ),
                                context=(
                                    current_trace.context
                                    if current_trace
                                    else None
                                ),
                                retrieval_context=(
                                    current_trace.retrieval_context
                                    if current_trace
                                    else None
                                ),
                                additional_metadata=golden.additional_metadata,
                                tools_called=(
                                    current_trace.tools_called
                                    if current_trace
                                    else None
                                ),
                                expected_tools=(
                                    current_trace.expected_tools
                                    if current_trace
                                    else None
                                ),
                                comments=golden.comments,
                                name=golden.name,
                                _dataset_alias=golden._dataset_alias,
                                _dataset_id=golden._dataset_id,
                            )

                        # Create a trace API if we have a trace
                        if trace_api is None and current_trace is not None:
                            trace_api = create_api_trace(current_trace, golden)

                        api_test_case = create_api_test_case(
                            test_case=test_case,
                            trace=trace_api,
                            index=count if not _is_assert_test else None,
                        )

                    if test_run is not None:
                        test_run_manager.set_test_run(test_run)

                    if api_test_case.success is None:
                        api_test_case.update_status(False)

                    # try to update metric data
                    if current_trace is not None:
                        if current_trace.metrics:
                            for m in current_trace.metrics:
                                if getattr(m, "skipped", False):
                                    continue
                                if id(m) in emitted_trace:
                                    continue
                                api_test_case.update_metric_data(
                                    create_metric_data(m)
                                )

                    # Finalize duration and persist
                    elapsed = time.perf_counter() - start_time
                    api_test_case.update_run_duration(
                        elapsed if elapsed >= 0 else deadline
                    )

                    if (
                        api_test_case.metrics_data == []
                        and api_test_case.trace is None
                    ):
                        api_test_case.metrics_data = None

                    test_run_manager.update_test_run(api_test_case, test_case)
                    test_results.append(create_test_result(api_test_case))

                    if trace_api is not None:
                        test_results.extend(
                            extract_trace_test_results(trace_api)
                        )

                    missing = 2 - pbar_case_increments
                    if missing > 0:
                        update_pbar(progress, pbar_id, advance=missing)

                finally:
                    # nothing to clean here, but keep symmetry with other paths
                    pass

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
                "Running Component-Level Evals (sync)",
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
    error_config: Optional[ErrorConfig] = ErrorConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:
    semaphore = asyncio.Semaphore(async_config.max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            timeout = _per_task_timeout()
            return await _await_with_outer_deadline(
                func, *args, timeout=timeout, **kwargs
            )

    test_run_manager = global_test_run_manager
    test_run_manager.save_to_disk = cache_config.write_cache
    test_run_manager.get_test_run(identifier=identifier)
    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True
    test_results: List[TestResult] = []
    tasks = []
    count = 0

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
                "Running Component-Level Evals (async)",
                total=len(goldens) * 2,
            )
            for golden in goldens:
                with capture_evaluation_run("golden"):
                    count += 1
                    task = execute_with_semaphore(
                        func=_a_execute_agentic_test_case,
                        golden=golden,
                        observed_callback=observed_callback,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=count,
                        verbose_mode=display_config.verbose_mode,
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
                    timeout=_gather_timeout(),
                )
            except (asyncio.TimeoutError, TimeoutError):
                # Cancel any still-pending tasks and drain them
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    else:
        for golden in goldens:
            with capture_evaluation_run("golden"):
                count += 1
                task = execute_with_semaphore(
                    func=_a_execute_agentic_test_case,
                    golden=golden,
                    observed_callback=observed_callback,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    count=count,
                    verbose_mode=display_config.verbose_mode,
                    ignore_errors=error_config.ignore_errors,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                    show_indicator=display_config.show_indicator,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                )
                tasks.append(asyncio.create_task(task))
                await asyncio.sleep(async_config.throttle_value)
        await asyncio.gather(*tasks)
    local_trace_manager.evaluating = False
    return test_results


async def _a_execute_agentic_test_case(
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
    trace_metrics: Optional[List[BaseMetric]] = None,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
    test_start_time = time.perf_counter()
    current_trace = None
    trace_api = None
    test_case = None
    api_test_case = None
    try:
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
                # get current_trace right away, we need it even if cancelled
                current_trace: Trace = current_trace_context.get()
                if asyncio.iscoroutinefunction(observed_callback):
                    await _await_with_outer_deadline(
                        observed_callback,
                        golden.input,
                        timeout=_per_task_timeout(),
                    )
                else:
                    observed_callback(golden.input)

            update_pbar(progress, pbar_tags_id, advance=total_tags)
            update_pbar(progress, pbar_id)

        elif trace:
            current_trace = trace

        trace_level_metrics_count = 0

        if trace_metrics:
            current_trace.metrics = trace_metrics

        # run evals through DFS
        trace_api = create_api_trace(trace=current_trace, golden=golden)

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

        await _a_execute_trace_test_case(
            trace=current_trace,
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

        async def dfs(trace: Trace, span: BaseSpan):
            await _a_execute_span_test_case(
                span=span,
                current_trace=trace,
                trace_api=trace_api,
                api_test_case=api_test_case,
                ignore_errors=ignore_errors,
                skip_on_missing_params=skip_on_missing_params,
                show_indicator=show_indicator,
                verbose_mode=verbose_mode,
                progress=progress,
                pbar_eval_id=pbar_eval_id,
                test_run_manager=test_run_manager,
                _use_bar_indicator=_use_bar_indicator,
            )

            if _skip_metrics_for_error(span=span, trace=trace):
                return

            child_tasks = [
                asyncio.create_task(dfs(trace, child))
                for child in span.children
            ]
            if child_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*child_tasks),
                        timeout=_gather_timeout(),
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    for t in child_tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*child_tasks, return_exceptions=True)
                    raise

        if not _skip_metrics_for_error(trace=current_trace):
            if current_trace and current_trace.root_spans:
                await dfs(current_trace, current_trace.root_spans[0])
            else:
                if (
                    logger.isEnabledFor(logging.DEBUG)
                    and get_settings().DEEPEVAL_VERBOSE_MODE
                ):
                    logger.debug(
                        "Skipping DFS: empty trace or no root spans (trace=%s)",
                        current_trace.uuid if current_trace else None,
                    )
    except asyncio.CancelledError:
        # mark any unfinished metrics as cancelled
        cancel_msg = (
            "Timed out/cancelled while evaluating agentic test case. "
            "Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set "
            "DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
        )

        if trace_metrics:
            for m in trace_metrics:
                if getattr(m, "skipped", False):
                    continue
                if getattr(m, "success", None) is None and not getattr(
                    m, "error", None
                ):
                    m.success = False
                    m.error = cancel_msg

        if trace is not None and trace.metrics:
            for m in trace.metrics:
                if getattr(m, "skipped", False):
                    continue
                if getattr(m, "success", None) is None and not getattr(
                    m, "error", None
                ):
                    m.success = False
                    m.error = cancel_msg
        if not ignore_errors:
            raise
    finally:
        try:
            if api_test_case is None:
                if test_case is None:
                    test_case = LLMTestCase(
                        input=golden.input,
                        actual_output=None,
                        expected_output=None,
                        context=None,
                        retrieval_context=None,
                        additional_metadata=golden.additional_metadata,
                        tools_called=None,
                        expected_tools=None,
                        comments=golden.comments,
                        name=golden.name,
                        _dataset_alias=golden._dataset_alias,
                        _dataset_id=golden._dataset_id,
                    )
                if trace is not None and trace_api is None:
                    trace_api = create_api_trace(trace, golden)

                api_test_case = create_api_test_case(
                    test_case=test_case,
                    trace=trace_api,
                    index=(count if not _is_assert_test else None),
                )

            # attach MetricData for any trace metrics we marked above
            if trace_metrics:
                for m in trace_metrics:
                    if getattr(m, "skipped", False):
                        continue
                    api_test_case.update_metric_data(create_metric_data(m))

            # If nothing set success yet, mark the case failed
            if api_test_case.success is None:
                api_test_case.update_status(False)

            # test_run_manager.update_test_run returns early if api_test_case.metrics_data is an empty list.
            # Set it to None to ensure the test_case is added
            if api_test_case.metrics_data == [] and api_test_case.trace is None:
                api_test_case.metrics_data = None

            # Duration & persist
            test_end_time = time.perf_counter()
            run_duration = test_end_time - test_start_time
            api_test_case.update_run_duration(run_duration)
            test_run_manager.update_test_run(api_test_case, test_case)

            # Build results and de-duplicate against trace results
            main_result = create_test_result(api_test_case)
            trace_results = (
                extract_trace_test_results(trace_api)
                if trace_api is not None
                else []
            )
            unique_trace_results = filter_duplicate_results(
                main_result, trace_results
            )
            test_results.append(main_result)
            test_results.extend(unique_trace_results)
            update_pbar(progress, pbar_id)
        finally:
            pass


async def _a_execute_span_test_case(
    span: BaseSpan,
    current_trace: Trace,
    trace_api: TraceApi,
    api_test_case: LLMApiTestCase,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    verbose_mode: Optional[bool],
    progress: Optional[Progress],
    pbar_eval_id: Optional[int],
    test_run_manager: Optional[TestRunManager],
    _use_bar_indicator: bool,
):
    api_span: BaseApiSpan = trace_manager._convert_span_to_api_span(span)
    if isinstance(span, AgentSpan):
        trace_api.agent_spans.append(api_span)
    elif isinstance(span, LlmSpan):
        trace_api.llm_spans.append(api_span)
        log_prompt(span, test_run_manager)
    elif isinstance(span, RetrieverSpan):
        trace_api.retriever_spans.append(api_span)
    elif isinstance(span, ToolSpan):
        trace_api.tool_spans.append(api_span)
    else:
        trace_api.base_spans.append(api_span)

    if _skip_metrics_for_error(span=span, trace=current_trace):
        api_span.status = TraceSpanApiStatus.ERRORED
        api_span.error = span.error or _trace_error(current_trace)
        if progress and pbar_eval_id is not None:
            update_pbar(
                progress,
                pbar_eval_id,
                advance=count_metrics_in_span_subtree(span),
            )
        return

    metrics: List[BaseMetric] = list(span.metrics or [])
    if not metrics:
        return

    requires_trace = any(metric.requires_trace for metric in metrics)

    llm_test_case = None
    if span.input:
        llm_test_case = LLMTestCase(
            input=str(span.input),
            actual_output=str(span.output) if span.output is not None else None,
            expected_output=span.expected_output,
            context=span.context,
            retrieval_context=span.retrieval_context,
            tools_called=span.tools_called,
            expected_tools=span.expected_tools,
        )

    if not requires_trace:
        if llm_test_case is None:
            api_span.status = TraceSpanApiStatus.ERRORED
            api_span.error = format_error_text(
                DeepEvalError(
                    "Span has metrics but no LLMTestCase. "
                    "Are you sure you called `update_current_span()`?"
                )
            )
            if progress and pbar_eval_id is not None:
                update_pbar(
                    progress,
                    pbar_eval_id,
                    advance=count_metrics_in_span_subtree(span),
                )
            return

    show_metrics_indicator = show_indicator and not _use_bar_indicator
    test_case: Optional[LLMTestCase] = llm_test_case

    # add trace if task completion
    if requires_trace:
        if test_case is None:
            test_case = LLMTestCase(input="None")
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


async def _a_execute_trace_test_case(
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

    if _skip_metrics_for_error(trace=trace):
        trace_api.status = TraceSpanApiStatus.ERRORED
        if progress and pbar_eval_id is not None:
            update_pbar(
                progress,
                pbar_eval_id,
                advance=count_total_metrics_for_trace(trace),
            )
        return

    metrics: List[BaseMetric] = list(trace.metrics or [])
    if not metrics:
        return

    requires_trace = any(metric.requires_trace for metric in metrics)

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

    if not requires_trace:
        if llm_test_case is None:
            trace.status = TraceSpanStatus.ERRORED
            trace_api.status = TraceSpanApiStatus.ERRORED
            if trace.root_spans:
                trace.root_spans[0].status = TraceSpanStatus.ERRORED
                trace.root_spans[0].error = format_error_text(
                    DeepEvalError(
                        "Trace has metrics but no LLMTestCase (missing input/output). "
                        "Are you sure you called `update_current_trace()`?"
                    )
                )
            if progress and pbar_eval_id is not None:
                update_pbar(
                    progress,
                    pbar_eval_id,
                    advance=count_total_metrics_for_trace(trace),
                )
            return

    show_metrics_indicator = show_indicator and not _use_bar_indicator
    test_case: Optional[LLMTestCase] = llm_test_case

    # add trace if task completion
    if requires_trace:
        if test_case is None:
            test_case = LLMTestCase(input="None")
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


###########################################
### Looped Evals
###########################################


def execute_agentic_test_cases_from_loop(
    goldens: List[Golden],
    trace_metrics: Optional[List[BaseMetric]],
    test_results: List[TestResult],
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> Iterator[TestResult]:

    test_run_manager = global_test_run_manager
    test_run_manager.save_to_disk = cache_config.write_cache
    test_run_manager.get_test_run(identifier=identifier)

    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True

    def evaluate_test_cases(
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> Iterator[Golden]:
        count = 0
        show_metric_indicator = (
            display_config.show_indicator and not _use_bar_indicator
        )

        for golden in goldens:
            token = set_current_golden(golden)
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
                    try:
                        # yield golden to user code
                        yield golden
                        # control has returned from user code without error, capture trace now
                        current_trace: Trace = current_trace_context.get()
                    finally:
                        # after user code returns control, always reset the context
                        reset_current_golden(token)

                update_pbar(progress, pbar_tags_id)
                update_pbar(progress, pbar_id)

                # Create empty trace api for llm api test case
                trace_api = create_api_trace(trace=current_trace, golden=golden)

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
                    metrics: List[BaseMetric] = list(span.metrics or [])

                    api_span: BaseApiSpan = (
                        trace_manager._convert_span_to_api_span(span)
                    )

                    if isinstance(span, AgentSpan):
                        trace_api.agent_spans.append(api_span)
                    elif isinstance(span, LlmSpan):
                        trace_api.llm_spans.append(api_span)
                        log_prompt(span, test_run_manager)
                    elif isinstance(span, RetrieverSpan):
                        trace_api.retriever_spans.append(api_span)
                    elif isinstance(span, ToolSpan):
                        trace_api.tool_spans.append(api_span)
                    else:
                        trace_api.base_spans.append(api_span)

                    # Skip errored trace/span
                    if _skip_metrics_for_error(span=span, trace=current_trace):
                        api_span.status = TraceSpanApiStatus.ERRORED
                        api_span.error = span.error or _trace_error(
                            current_trace
                        )
                        if progress and pbar_eval_id is not None:
                            update_pbar(
                                progress,
                                pbar_eval_id,
                                advance=count_metrics_in_span_subtree(span),
                            )
                        return

                    for child in span.children:
                        dfs(child, progress, pbar_eval_id)

                    if not span.metrics:
                        return

                    requires_trace = any(
                        metric.requires_trace for metric in metrics
                    )

                    llm_test_case = None
                    if span.input is not None:
                        llm_test_case = LLMTestCase(
                            input=str(span.input),
                            actual_output=(
                                str(span.output)
                                if span.output is not None
                                else None
                            ),
                            expected_output=span.expected_output,
                            context=span.context,
                            retrieval_context=span.retrieval_context,
                            tools_called=span.tools_called,
                            expected_tools=span.expected_tools,
                        )

                    if requires_trace:
                        if llm_test_case is None:
                            llm_test_case = LLMTestCase(input="None")
                        llm_test_case._trace_dict = (
                            trace_manager.create_nested_spans_dict(span)
                        )
                    else:
                        if llm_test_case is None:
                            api_span.status = TraceSpanApiStatus.ERRORED
                            api_span.error = format_error_text(
                                DeepEvalError(
                                    "Span has metrics but no LLMTestCase. "
                                    "Are you sure you called `update_current_span()`?"
                                )
                            )
                            if progress and pbar_eval_id is not None:
                                update_pbar(
                                    progress,
                                    pbar_eval_id,
                                    advance=count_metrics_in_span_subtree(span),
                                )
                            return

                    # Preparing metric calculation
                    api_span.metrics_data = []
                    for metric in metrics:
                        metric.skipped = False
                        metric.error = None
                        if display_config.verbose_mode is not None:
                            metric.verbose_mode = display_config.verbose_mode

                    # Metric calculation
                    for metric in metrics:
                        metric_data = None
                        res = _execute_metric(
                            metric=metric,
                            test_case=llm_test_case,
                            show_metric_indicator=show_metric_indicator,
                            in_component=True,
                            error_config=error_config,
                        )
                        if res == "skip":
                            continue

                        metric_data = create_metric_data(metric)
                        api_span.metrics_data.append(metric_data)
                        api_test_case.update_status(metric_data.success)
                        update_pbar(progress, pbar_eval_id)

                if trace_metrics:
                    current_trace.metrics = trace_metrics

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
                skip_metrics_for_this_golden = False
                if _skip_metrics_for_error(trace=current_trace):
                    trace_api.status = TraceSpanApiStatus.ERRORED
                    if progress and pbar_eval_id is not None:
                        update_pbar(
                            progress,
                            pbar_eval_id,
                            advance=count_total_metrics_for_trace(
                                current_trace
                            ),
                        )
                else:
                    if current_trace.metrics:
                        requires_trace = any(
                            metric.requires_trace
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

                        if requires_trace:
                            if llm_test_case is None:
                                llm_test_case = LLMTestCase(input="None")
                            llm_test_case._trace_dict = (
                                trace_manager.create_nested_spans_dict(
                                    current_trace.root_spans[0]
                                )
                            )
                        else:
                            if llm_test_case is None:
                                current_trace.status = TraceSpanStatus.ERRORED
                                trace_api.status = TraceSpanApiStatus.ERRORED
                                if current_trace.root_spans:
                                    current_trace.root_spans[0].status = (
                                        TraceSpanStatus.ERRORED
                                    )
                                    current_trace.root_spans[0].error = (
                                        format_error_text(
                                            DeepEvalError(
                                                "Trace has metrics but no LLMTestCase (missing input/output). "
                                                "Are you sure you called `update_current_trace()`?"
                                            )
                                        )
                                    )
                                if progress and pbar_eval_id is not None:
                                    update_pbar(
                                        progress,
                                        pbar_eval_id,
                                        advance=count_total_metrics_for_trace(
                                            current_trace
                                        ),
                                    )
                                skip_metrics_for_this_golden = True

                        if not skip_metrics_for_this_golden:
                            for metric in current_trace.metrics:
                                metric.skipped = False
                                metric.error = None
                                if display_config.verbose_mode is not None:
                                    metric.verbose_mode = (
                                        display_config.verbose_mode
                                    )

                            trace_api.metrics_data = []
                            for metric in current_trace.metrics:
                                res = _execute_metric(
                                    metric=metric,
                                    test_case=llm_test_case,
                                    show_metric_indicator=show_metric_indicator,
                                    in_component=True,
                                    error_config=error_config,
                                )
                                if res == "skip":
                                    continue

                                if not metric.skipped:
                                    metric_data = create_metric_data(metric)
                                    trace_api.metrics_data.append(metric_data)
                                    api_test_case.update_metric_data(
                                        metric_data
                                    )
                                    api_test_case.update_status(
                                        metric_data.success
                                    )
                                    update_pbar(progress, pbar_eval_id)

                    # Then handle span-level metrics
                    dfs(current_trace.root_spans[0], progress, pbar_eval_id)

            end_time = time.perf_counter()
            run_duration = end_time - start_time
            # Update test run
            api_test_case.update_run_duration(run_duration)
            test_run_manager.update_test_run(api_test_case, test_case)
            main_result = create_test_result(api_test_case)
            trace_results = extract_trace_test_results(trace_api)
            unique_trace_results = filter_duplicate_results(
                main_result, trace_results
            )
            test_results.append(main_result)
            test_results.extend(unique_trace_results)

            update_pbar(progress, pbar_id)

    try:
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
                    "Running Component-Level Evals (sync)",
                    total=len(goldens) * 2,
                )
                yield from evaluate_test_cases(
                    progress=progress, pbar_id=pbar_id
                )
        else:
            yield from evaluate_test_cases()
    except Exception:
        raise
    finally:
        local_trace_manager.evaluating = False
        local_trace_manager.traces_to_evaluate_order.clear()
        local_trace_manager.traces_to_evaluate.clear()
        local_trace_manager.trace_uuid_to_golden.clear()


def a_execute_agentic_test_cases_from_loop(
    goldens: List[Golden],
    trace_metrics: Optional[List[BaseMetric]],
    test_results: List[TestResult],
    loop: asyncio.AbstractEventLoop,
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    cache_config: Optional[CacheConfig] = CacheConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    identifier: Optional[str] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> Iterator[TestResult]:

    semaphore = asyncio.Semaphore(async_config.max_concurrent)
    original_create_task = asyncio.create_task

    test_run_manager = global_test_run_manager
    test_run_manager.save_to_disk = cache_config.write_cache
    test_run = test_run_manager.get_test_run(identifier=identifier)

    local_trace_manager = trace_manager
    local_trace_manager.evaluating = True
    local_trace_manager.evaluation_loop = True

    async def execute_callback_with_semaphore(coroutine: Awaitable):
        async with semaphore:
            timeout = _per_task_timeout()
            return await _await_with_outer_deadline(coroutine, timeout=timeout)

    def evaluate_test_cases(
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
        pbar_callback_id: Optional[int] = None,
    ):
        # Tasks we scheduled during this iterator run on this event loop.
        # by gathering these tasks we can avoid re-awaiting coroutines which
        # can cause cross loop mixups that trigger "future belongs to a different loop" errors
        created_tasks: list[asyncio.Task] = []
        task_meta: dict[asyncio.Task, dict] = {}
        current_golden_ctx = {"index": -1, "name": None, "input": None}

        def create_callback_task(coro, **kwargs):
            # build a descriptive task name for tracking
            coro_desc = repr(coro)
            task_name = f"callback[{current_golden_ctx['index']}]:{coro_desc.split()[1] if ' ' in coro_desc else coro_desc}"

            # Wrap the user coroutine in our semaphore runner and bind it to THIS loop.
            # Keep the resulting Task so we can gather tasks (not raw coroutines) later,
            # without touching tasks from other loops or already awaited coroutines.
            task = loop.create_task(
                execute_callback_with_semaphore(coro), name=task_name
            )

            # record metadata for debugging
            started = time.perf_counter()
            short_input = current_golden_ctx.get("input")
            if isinstance(short_input, str):
                short_input = shorten(short_input, len_medium())

            task_meta[task] = {
                "golden_index": current_golden_ctx["index"],
                "golden_name": current_golden_ctx["name"],
                "input": short_input,
                "coro": coro_desc,
                "started": started,
            }

            def on_task_done(t: asyncio.Task):
                cancelled = False
                exc = None
                trace = None
                root = None
                resolved_trace_from_task = False
                resolved_root_from_task = False

                # Task.exception() raises CancelledError if task was cancelled
                try:
                    exc = t.exception()
                except asyncio.CancelledError:
                    cancelled = True
                    exc = None

                meta = task_meta.get(t, {})
                golden_index = meta.get("golden_index")

                if golden_index is not None and 0 <= golden_index < len(
                    goldens
                ):
                    golden = goldens[golden_index]

                    def _mark_trace_error(trace, root, msg: str):
                        now = time.perf_counter()
                        trace.status = TraceSpanStatus.ERRORED
                        # Close the trace so the API layer has a proper endTime
                        if trace.end_time is None:
                            trace.end_time = now
                        if root:
                            root.status = TraceSpanStatus.ERRORED
                            root.error = msg
                            if root.end_time is None:
                                root.end_time = now

                    if exc is not None:
                        msg = format_error_text(exc)
                        trace, root = _resolve_trace_and_root_for_task(t)
                        resolved_trace_from_task = bool(trace)
                        resolved_root_from_task = bool(root)
                        if trace:
                            _mark_trace_error(trace, root, msg)
                        else:
                            for (
                                trace
                            ) in trace_manager.integration_traces_to_evaluate:
                                if (
                                    trace_manager.trace_uuid_to_golden.get(
                                        trace.uuid
                                    )
                                    is golden
                                ):
                                    root = _pick_root_for_marking(trace)
                                    _mark_trace_error(trace, root, msg)
                                    break

                    elif cancelled or t.cancelled():
                        cancel_exc = DeepEvalError(
                            "Task was cancelled (likely due to timeout)."
                        )
                        msg = format_error_text(cancel_exc)
                        trace, root = _resolve_trace_and_root_for_task(t)
                        resolved_trace_from_task = bool(trace)
                        resolved_root_from_task = bool(root)
                        if trace:
                            _mark_trace_error(trace, root, msg)
                        else:
                            for (
                                trace
                            ) in trace_manager.integration_traces_to_evaluate:
                                if (
                                    trace_manager.trace_uuid_to_golden.get(
                                        trace.uuid
                                    )
                                    is golden
                                ):
                                    root = _pick_root_for_marking(trace)
                                    _mark_trace_error(trace, root, msg)
                                    break

                if get_settings().DEEPEVAL_DEBUG_ASYNC:
                    # Using info level here to make it easy to spot these logs.
                    golden_name = meta.get("golden_name")
                    duration = time.perf_counter() - meta.get(
                        "started", started
                    )

                    if cancelled or exc is not None:
                        if not resolved_trace_from_task:
                            logger.warning(
                                "[deepeval] on_task_done: no binding for task; falling back to golden->trace. task=%s golden=%r",
                                t.get_name(),
                                golden_name,
                            )
                        elif not resolved_root_from_task:
                            logger.warning(
                                "[deepeval] on_task_done: bound trace found but no bound root; using heuristic. task=%s trace=%s",
                                t.get_name(),
                                trace.uuid,
                            )

                    if cancelled:
                        logger.info(
                            "[deepeval] task CANCELLED %s after %.2fs meta=%r",
                            t.get_name(),
                            duration,
                            meta,
                        )
                    elif exc is not None:

                        show_trace = bool(
                            get_settings().DEEPEVAL_LOG_STACK_TRACES
                        )
                        exc_info = (
                            (
                                type(exc),
                                exc,
                                getattr(exc, "__traceback__", None),
                            )
                            if show_trace
                            else None
                        )
                        logger.error(
                            "[deepeval] task ERROR %s after %.2fs meta=%r",
                            t.get_name(),
                            duration,
                            meta,
                            exc_info=exc_info,
                        )
                    else:
                        logger.info(
                            "[deepeval] task OK %s after %.2fs meta={'golden_index': %r}",
                            t.get_name(),
                            duration,
                            meta.get("golden_index"),
                        )

                try:
                    trace_manager.task_bindings.pop(t, None)
                except Exception:
                    pass
                update_pbar(progress, pbar_callback_id)
                update_pbar(progress, pbar_id)

            task.add_done_callback(on_task_done)
            created_tasks.append(task)
            return task

        asyncio.create_task = create_callback_task
        # DEBUG
        # Snapshot tasks that already exist on this loop so we can detect strays
        baseline_tasks = loop.run_until_complete(_snapshot_tasks())

        try:
            for index, golden in enumerate(goldens):
                token = set_current_golden(golden)
                current_golden_ctx.update(
                    {
                        "index": index,
                        "name": getattr(golden, "name", None),
                        "input": getattr(golden, "input", None),
                    }
                )
                prev_task_length = len(created_tasks)
                try:
                    yield golden
                finally:
                    reset_current_golden(token)
                # if this golden created no tasks, bump bars now
                if len(created_tasks) == prev_task_length:
                    update_pbar(progress, pbar_callback_id)
                    update_pbar(progress, pbar_id)
        finally:
            asyncio.create_task = original_create_task

        if created_tasks:
            # Only await tasks we created on this loop in this run.
            # This will prevent re-awaiting and avoids cross loop "future belongs to a different loop" errors
            try:
                loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.gather(*created_tasks, return_exceptions=True),
                        timeout=_gather_timeout(),
                    )
                )

            except (asyncio.TimeoutError, TimeoutError):
                import traceback

                pending = [t for t in created_tasks if not t.done()]

                # Log the elapsed time for each task that was pending
                for t in pending:
                    meta = task_meta.get(t, {})
                    start_time = meta.get("started", time.perf_counter())
                    elapsed_time = time.perf_counter() - start_time

                    # Determine if it was a per task or gather timeout based on task's elapsed time
                    if elapsed_time >= _per_task_timeout():
                        timeout_type = "per-task"
                    else:
                        timeout_type = "gather"

                    logger.warning(
                        f"[deepeval] gather TIMEOUT after {_gather_timeout()}s; "
                        f"pending={len(pending)} tasks. Timeout type: {timeout_type}. "
                        f"To give tasks more time, consider increasing "
                        f"DEEPEVAL_PER_TASK_TIMEOUT_SECONDS for longer task completion time or "
                        f"DEEPEVAL_TASK_GATHER_BUFFER_SECONDS to allow more time for gathering results."
                    )

                    # Log pending tasks and their stack traces
                    logger.info(
                        "  - PENDING %s elapsed_time=%.2fs meta=%s",
                        t.get_name(),
                        elapsed_time,
                        meta,
                    )
                    if loop.get_debug() and get_settings().DEEPEVAL_DEBUG_ASYNC:
                        frames = t.get_stack(limit=6)
                        if frames:
                            logger.info("    stack:")
                            for fr in frames:
                                for line in traceback.format_stack(fr):
                                    logger.info("      " + line.rstrip())

                # Cancel and drain the tasks
                for t in pending:
                    t.cancel()
                loop.run_until_complete(
                    asyncio.gather(*created_tasks, return_exceptions=True)
                )
            finally:

                # if it is already closed, we are done
                if loop.is_closed():
                    return

                try:
                    current_tasks = set()
                    # Find tasks that were created during this run but we didn’t track
                    current_tasks = loop.run_until_complete(_snapshot_tasks())
                except RuntimeError:
                    # this might happen if the loop is already closing
                    pass

                leftovers = [
                    t
                    for t in current_tasks
                    if t not in baseline_tasks
                    and t not in created_tasks
                    and not t.done()
                ]

                if get_settings().DEEPEVAL_DEBUG_ASYNC:
                    if len(leftovers) > 0:
                        logger.warning(
                            "[deepeval] %d stray task(s) not tracked; cancelling...",
                            len(leftovers),
                        )
                    for t in leftovers:
                        meta = task_meta.get(t, {})
                        name = t.get_name()
                        logger.warning("  - STRAY %s meta=%s", name, meta)

                if leftovers:
                    for t in leftovers:
                        t.cancel()

                    # Drain strays so they don’t leak into the next iteration
                    try:
                        loop.run_until_complete(
                            asyncio.gather(*leftovers, return_exceptions=True)
                        )
                    except RuntimeError:
                        # If the loop is closing here, just continue
                        if get_settings().DEEPEVAL_DEBUG_ASYNC:
                            logger.warning(
                                "[deepeval] failed to drain stray tasks because loop is closing"
                            )

        # Evaluate traces
        if trace_manager.traces_to_evaluate:
            loop.run_until_complete(
                _a_evaluate_traces(
                    traces_to_evaluate=trace_manager.traces_to_evaluate,
                    goldens=goldens,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    trace_metrics=trace_metrics,
                    verbose_mode=display_config.verbose_mode,
                    ignore_errors=error_config.ignore_errors,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                    show_indicator=display_config.show_indicator,
                    throttle_value=async_config.throttle_value,
                    max_concurrent=async_config.max_concurrent,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                    progress=progress,
                    pbar_id=pbar_id,
                )
            )
        elif trace_manager.integration_traces_to_evaluate:
            loop.run_until_complete(
                _a_evaluate_traces(
                    traces_to_evaluate=trace_manager.integration_traces_to_evaluate,
                    goldens=goldens,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    trace_metrics=trace_metrics,
                    verbose_mode=display_config.verbose_mode,
                    ignore_errors=error_config.ignore_errors,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                    show_indicator=display_config.show_indicator,
                    throttle_value=async_config.throttle_value,
                    max_concurrent=async_config.max_concurrent,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                    progress=progress,
                    pbar_id=pbar_id,
                )
            )
        elif trace_manager.test_case_metrics:
            loop.run_until_complete(
                _evaluate_test_case_pairs(
                    test_case_pairs=trace_manager.test_case_metrics,
                    test_run=test_run,
                    test_run_manager=test_run_manager,
                    test_results=test_results,
                    ignore_errors=error_config.ignore_errors,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                    show_indicator=display_config.show_indicator,
                    verbose_mode=display_config.verbose_mode,
                    throttle_value=async_config.throttle_value,
                    max_concurrent=async_config.max_concurrent,
                    _use_bar_indicator=_use_bar_indicator,
                    _is_assert_test=_is_assert_test,
                    progress=progress,
                    pbar_id=pbar_id,
                )
            )

    try:
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
                    "Running Component-Level Evals (async)",
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
    except Exception:
        raise
    finally:
        local_trace_manager.evaluating = False
        local_trace_manager.traces_to_evaluate_order.clear()
        local_trace_manager.traces_to_evaluate.clear()
        local_trace_manager.trace_uuid_to_golden.clear()


async def _a_evaluate_traces(
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
    trace_metrics: Optional[List[BaseMetric]],
):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_evals_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            timeout = _per_task_timeout()
            return await _await_with_outer_deadline(
                func, *args, timeout=timeout, **kwargs
            )

    eval_tasks = []
    # Here, we will work off a fixed-set copy to avoid surprises from potential
    # mid-iteration mutation
    traces_snapshot = list(traces_to_evaluate or [])

    for count, trace in enumerate(traces_snapshot):
        # Prefer the explicit mapping from trace -> golden captured at trace creation.
        golden = trace_manager.trace_uuid_to_golden.get(trace.uuid)
        if not golden:
            # trace started during evaluation_loop but the CURRENT_GOLDEN was
            # not set for some reason. We can’t map it to a golden, so the best
            # we can do is skip evaluation for this trace.
            if (
                logger.isEnabledFor(logging.DEBUG)
                and get_settings().DEEPEVAL_VERBOSE_MODE
            ):
                logger.debug(
                    "Skipping trace %s: no golden association found during evaluation_loop ",
                    trace.uuid,
                )
            continue
        with capture_evaluation_run("golden"):
            task = execute_evals_with_semaphore(
                func=_a_execute_agentic_test_case,
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
                trace_metrics=trace_metrics,
            )
            eval_tasks.append(asyncio.create_task(task))
            await asyncio.sleep(throttle_value)

    try:
        await asyncio.wait_for(
            asyncio.gather(*eval_tasks),
            timeout=_gather_timeout(),
        )
    except (asyncio.TimeoutError, TimeoutError):
        for t in eval_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*eval_tasks, return_exceptions=True)
        raise


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
            timeout = _per_task_timeout()
            return await _await_with_outer_deadline(
                func, *args, timeout=timeout, **kwargs
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
            timeout=_gather_timeout(),
        )
    except (asyncio.TimeoutError, TimeoutError):
        # Cancel any still-pending tasks and drain them
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _execute_metric(
    metric: BaseMetric,
    test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    show_metric_indicator: bool,
    in_component: bool,
    error_config: ErrorConfig,
) -> Optional[str]:
    try:
        metric.measure(
            test_case,
            _show_indicator=show_metric_indicator,
            _in_component=in_component,
            _log_metric_to_confident=False,
        )
    except MissingTestCaseParamsError as e:
        if error_config.skip_on_missing_params:
            metric.skipped = True
            metric.error = None
            metric.success = None
            return "skip"
        else:
            if error_config.ignore_errors:
                metric.error = format_error_text(e)
                metric.success = False
            else:
                raise
    except TypeError:
        try:
            metric.measure(test_case)
        except MissingTestCaseParamsError as e:
            if error_config.skip_on_missing_params:
                metric.skipped = True
                metric.error = None
                metric.success = None
                return "skip"
            else:
                if error_config.ignore_errors:
                    metric.error = format_error_text(e)
                    metric.success = False
                else:
                    raise
        except Exception as e:
            if error_config.ignore_errors:
                metric.error = format_error_text(e)
                metric.success = False
            else:
                raise
    except Exception as e:
        if error_config.ignore_errors:
            metric.error = format_error_text(e)
            metric.success = False
        else:
            raise


def log_prompt(
    llm_span: LlmSpan,
    test_run_manager: TestRunManager,
):
    prompt = llm_span.prompt
    if prompt is None:
        return

    span_hyperparameters = {}
    prompt_version = prompt.version if is_confident() else None
    key = f"{prompt.alias}_{prompt_version}"
    span_hyperparameters[key] = prompt

    test_run = test_run_manager.get_test_run()
    if test_run.prompts is None:
        test_run.prompts = []
    if test_run.hyperparameters is None:
        test_run.hyperparameters = {}

    if key not in test_run.hyperparameters:
        test_run.hyperparameters.update(
            process_hyperparameters(span_hyperparameters, False)
        )
        existing_prompt_keys = {
            f"{p.alias}_{p.version}" for p in test_run.prompts
        }
        new_prompts = process_prompts(span_hyperparameters)
        for new_prompt in new_prompts:
            new_prompt_key = f"{new_prompt.alias}_{new_prompt.version}"
            if new_prompt_key not in existing_prompt_keys:
                test_run.prompts.append(new_prompt)

    global_test_run_manager.save_test_run(TEMP_FILE_PATH)
