import logging

from typing import (
    List,
    Optional,
)
import time

from deepeval.evaluate.configs import (
    ErrorConfig,
    DisplayConfig,
)
from deepeval.tracing.tracing import (
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
from deepeval.errors import DeepEvalError
from deepeval.utils import (
    format_error_text,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
)
from deepeval.test_case.api import create_api_test_case
from deepeval.test_run import (
    global_test_run_manager,
)
from deepeval.evaluate.types import TestResult
from deepeval.evaluate.utils import (
    create_api_trace,
    create_metric_data,
    create_test_result,
)
from deepeval.tracing.types import TraceSpanStatus
from deepeval.tracing.api import TraceSpanApiStatus
from deepeval.test_run import TEMP_FILE_PATH

logger = logging.getLogger(__name__)


from deepeval.evaluate.execute._common import (
    _execute_metric,
    _skip_metrics_for_error,
    _trace_error,
    log_prompt,
)


def _assert_test_from_current_trace(
    golden: Golden,
    metrics: Optional[List[BaseMetric]] = None,
    error_config: Optional[ErrorConfig] = None,
    display_config: Optional[DisplayConfig] = None,
) -> TestResult:
    """Attach the test's live `@observe` trace to the active test run.

    Relies on the deepeval pytest plugin's eval scope to keep the trace live
    across the test body so it can be read off `current_trace_context` here.
    """
    if error_config is None:
        error_config = ErrorConfig()
    if display_config is None:
        display_config = DisplayConfig(show_indicator=False)

    current_trace: Optional[Trace] = current_trace_context.get()
    if current_trace is None:
        raise DeepEvalError(
            "No active trace found for this test. "
            "`assert_test(golden=..., metrics=...)` must be called inside a "
            "pytest test run with `deepeval test run`, and the test body must "
            "invoke at least one `@observe`-decorated function."
        )

    test_run_manager = global_test_run_manager

    # Trace is mid-flight (outer wrapper span hasn't closed); stamp end_time.
    if current_trace.end_time is None:
        current_trace.end_time = time.perf_counter()

    # Mirror native Observer behavior: trace errors only if the user's root
    # span errors. Nested errors caught by user code don't taint the trace.
    user_roots: List[BaseSpan] = []
    for s in current_trace.root_spans or []:
        if getattr(s, "name", None) == "Test Wrapper" and s.children:
            user_roots.extend(s.children)
        else:
            user_roots.append(s)
    errored = any(s.status == TraceSpanStatus.ERRORED for s in user_roots)
    current_trace.status = (
        TraceSpanStatus.ERRORED if errored else TraceSpanStatus.SUCCESS
    )

    trace_api = create_api_trace(trace=current_trace, golden=golden)
    trace_api.status = (
        TraceSpanApiStatus.ERRORED if errored else TraceSpanApiStatus.SUCCESS
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
        additional_metadata=golden.additional_metadata,
        tools_called=current_trace.tools_called,
        expected_tools=current_trace.expected_tools,
        comments=golden.comments,
        name=golden.name,
        _dataset_alias=golden._dataset_alias,
        _dataset_id=golden._dataset_id,
        _dataset_rank=golden._dataset_rank,
    )
    api_test_case = create_api_test_case(
        test_case=test_case,
        trace=trace_api,
        index=None,
    )

    def dfs(span: BaseSpan, is_promoted_root: bool = False):
        metrics: List[BaseMetric] = list(span.metrics or [])
        api_span: BaseApiSpan = trace_manager._convert_span_to_api_span(span)

        # Promoted root's parent_uuid still points at the stripped wrapper;
        # null it so the backend treats it as a genuine root.
        if is_promoted_root:
            api_span.parent_uuid = None

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
            return

        for child in span.children:
            dfs(child)

        if not metrics:
            return

        requires_trace = any(
            getattr(m, "requires_trace", False) for m in metrics
        )

        llm_test_case: Optional[LLMTestCase] = None
        if span.input is not None:
            llm_test_case = LLMTestCase(
                input=str(span.input),
                actual_output=(
                    str(span.output) if span.output is not None else None
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
            llm_test_case._trace_dict = trace_manager.create_nested_spans_dict(
                span
            )
        elif llm_test_case is None:
            api_span.status = TraceSpanApiStatus.ERRORED
            api_span.error = format_error_text(
                DeepEvalError(
                    "Span has metrics but no LLMTestCase. "
                    "Are you sure you called `update_current_span()`?"
                )
            )
            return

        api_span.metrics_data = []
        for metric in metrics:
            metric.skipped = False
            metric.error = None
            if display_config.verbose_mode is not None:
                metric.verbose_mode = display_config.verbose_mode

        for metric in metrics:
            res = _execute_metric(
                metric=metric,
                test_case=llm_test_case,
                show_metric_indicator=False,
                in_component=True,
                error_config=error_config,
            )
            if res == "skip":
                continue
            metric_data = create_metric_data(metric)
            api_span.metrics_data.append(metric_data)
            api_test_case.update_status(metric_data.success)

    # Skip the plugin's "Test Wrapper" and promote its first child as the root.
    root_for_dfs: Optional[BaseSpan] = None
    is_promoted_root = False
    if current_trace.root_spans:
        root = current_trace.root_spans[0]
        if getattr(root, "name", None) == "Test Wrapper" and root.children:
            root_for_dfs = root.children[0]
            is_promoted_root = True
        else:
            root_for_dfs = root

    if root_for_dfs is not None:
        dfs(root_for_dfs, is_promoted_root=is_promoted_root)

    existing_trace_metrics = list(current_trace.metrics or [])
    if metrics:
        existing_trace_metrics = existing_trace_metrics + list(metrics)
    current_trace.metrics = existing_trace_metrics

    if current_trace.metrics and not _skip_metrics_for_error(
        trace=current_trace
    ):
        llm_test_case_for_trace = LLMTestCase(
            input=(
                str(current_trace.input)
                if current_trace.input is not None
                else golden.input or "None"
            ),
            actual_output=(
                str(current_trace.output)
                if current_trace.output is not None
                else None
            ),
            expected_output=current_trace.expected_output
            or golden.expected_output,
            context=current_trace.context or golden.context,
            retrieval_context=current_trace.retrieval_context
            or golden.retrieval_context,
            tools_called=current_trace.tools_called,
            expected_tools=current_trace.expected_tools
            or golden.expected_tools,
        )
        if (
            any(
                getattr(m, "requires_trace", False)
                for m in current_trace.metrics
            )
            and root_for_dfs is not None
        ):
            llm_test_case_for_trace._trace_dict = (
                trace_manager.create_nested_spans_dict(root_for_dfs)
            )

        trace_api.metrics_data = []
        for metric in current_trace.metrics:
            metric.skipped = False
            metric.error = None
            if display_config.verbose_mode is not None:
                metric.verbose_mode = display_config.verbose_mode

            res = _execute_metric(
                metric=metric,
                test_case=llm_test_case_for_trace,
                show_metric_indicator=False,
                in_component=True,
                error_config=error_config,
            )
            if res == "skip":
                continue
            if not metric.skipped:
                metric_data = create_metric_data(metric)
                trace_api.metrics_data.append(metric_data)
                api_test_case.update_metric_data(metric_data)
                api_test_case.update_status(metric_data.success)

    test_run_manager.update_test_run(api_test_case, test_case)
    test_run_manager.save_test_run(TEMP_FILE_PATH)

    return create_test_result(api_test_case)
