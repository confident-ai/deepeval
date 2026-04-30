import logging

from rich.progress import (
    Progress,
)
from typing import (
    List,
    Optional,
    Union,
)
import asyncio
import time


from deepeval.tracing.tracing import (
    trace_manager,
    Trace,
    BaseSpan,
    AgentSpan,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
)
from deepeval.tracing.api import (
    TraceApi,
    BaseApiSpan,
)
from deepeval.dataset import Golden
from deepeval.errors import DeepEvalError
from deepeval.utils import (
    format_error_text,
    get_gather_timeout,
)
from deepeval.metrics import (
    BaseMetric,
)
from deepeval.metrics.indicator import (
    measure_metrics_with_indicator,
)
from deepeval.test_case import (
    LLMTestCase,
)
from deepeval.test_case.api import create_api_test_case
from deepeval.test_run import (
    LLMApiTestCase,
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
from deepeval.utils import add_pbar, update_pbar
from deepeval.tracing.types import TraceSpanStatus
from deepeval.tracing.api import TraceSpanApiStatus
from deepeval.config.settings import get_settings

logger = logging.getLogger(__name__)


from deepeval.evaluate.execute._common import (
    _skip_metrics_for_error,
    _trace_error,
    filter_duplicate_results,
    log_prompt,
)


async def _a_execute_agentic_test_case(
    golden: Golden,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, LLMTestCase]],
    count: int,
    verbose_mode: Optional[bool],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    trace: Trace,
    trace_metrics: Optional[List[BaseMetric]] = None,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
    test_start_time = time.perf_counter()
    current_trace: Trace = trace
    trace_api = None
    test_case = None
    api_test_case = None
    try:
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
            metadata=golden.additional_metadata,
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
                        timeout=get_gather_timeout(),
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
        if get_settings().DEEPEVAL_DISABLE_TIMEOUTS:
            cancel_msg = (
                "Cancelled while evaluating agentic test case. "
                "(DeepEval timeouts are disabled; this cancellation likely came from upstream orchestration or manual cancellation). "
                "Set DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
            )
        else:
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
                        metadata=golden.additional_metadata,
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
