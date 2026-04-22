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
    Awaitable,
    Iterator,
)
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
    BaseApiSpan,
)
from deepeval.dataset import Golden
from deepeval.contextvars import set_current_golden, reset_current_golden
from deepeval.errors import DeepEvalError
from deepeval.metrics.utils import copy_metrics
from deepeval.utils import (
    shorten,
    len_medium,
    format_error_text,
    get_per_task_timeout_seconds,
    get_gather_timeout,
)
from deepeval.telemetry import capture_evaluation_run
from deepeval.metrics import BaseMetric

from deepeval.test_case import (
    LLMTestCase,
)
from deepeval.test_case.api import create_api_test_case
from deepeval.test_run import (
    global_test_run_manager,
    TestRunManager,
)
from deepeval.evaluate.types import TestResult
from deepeval.evaluate.utils import (
    create_api_trace,
    create_metric_data,
    create_test_result,
    count_metrics_in_trace,
    count_total_metrics_for_trace,
    count_metrics_in_span_subtree,
    extract_trace_test_results,
)
from deepeval.utils import add_pbar, update_pbar, custom_console
from deepeval.tracing.types import TraceSpanStatus
from deepeval.tracing.api import TraceSpanApiStatus
from deepeval.config.settings import get_settings

logger = logging.getLogger(__name__)


from deepeval.evaluate.execute._common import (
    _await_with_outer_deadline,
    _execute_metric,
    _log_gather_timeout,
    _pick_root_for_marking,
    _resolve_trace_and_root_for_task,
    _skip_metrics_for_error,
    _snapshot_tasks,
    _trace_error,
    filter_duplicate_results,
    log_prompt,
)
from deepeval.evaluate.execute.agentic import (
    _a_execute_agentic_test_case,
)
from deepeval.evaluate.execute.e2e import _evaluate_test_case_pairs


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
            return await _await_with_outer_deadline(
                coroutine, timeout=get_per_task_timeout_seconds()
            )

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
                        timeout=get_gather_timeout(),
                    )
                )

            except (asyncio.TimeoutError, TimeoutError) as e:
                import traceback

                settings = get_settings()
                pending = [t for t in created_tasks if not t.done()]

                _log_gather_timeout(logger, exc=e, pending=len(pending))

                # Log the elapsed time for each task that was pending
                for t in pending:
                    meta = task_meta.get(t, {})
                    start_time = meta.get("started", time.perf_counter())
                    elapsed_time = time.perf_counter() - start_time

                    # Determine if it was a per task or gather timeout based on task's elapsed time
                    if not settings.DEEPEVAL_DISABLE_TIMEOUTS:
                        timeout_type = (
                            "per-task"
                            if elapsed_time >= get_per_task_timeout_seconds()
                            else "gather"
                        )
                        logger.info(
                            "  - PENDING %s elapsed_time=%.2fs timeout_type=%s meta=%s",
                            t.get_name(),
                            elapsed_time,
                            timeout_type,
                            meta,
                        )
                    else:
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
            return await _await_with_outer_deadline(
                func, *args, timeout=get_per_task_timeout_seconds(), **kwargs
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
        copied_trace_metrics: Optional[List[BaseMetric]] = None
        if trace_metrics:
            copied_trace_metrics = copy_metrics(trace_metrics)
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
                trace_metrics=copied_trace_metrics,
            )
            eval_tasks.append(asyncio.create_task(task))
            await asyncio.sleep(throttle_value)

    try:
        await asyncio.wait_for(
            asyncio.gather(*eval_tasks),
            timeout=get_gather_timeout(),
        )
    except (asyncio.TimeoutError, TimeoutError):
        for t in eval_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*eval_tasks, return_exceptions=True)
        raise
