import inspect
import logging

from typing import (
    List,
    Optional,
    Union,
)
import asyncio

from deepeval.evaluate.configs import (
    ErrorConfig,
)
from deepeval.tracing.tracing import (
    trace_manager,
    Trace,
    BaseSpan,
    LlmSpan,
)
from deepeval.errors import MissingTestCaseParamsError
from deepeval.utils import (
    format_error_text,
    are_timeouts_disabled,
    get_gather_timeout_seconds,
)
from deepeval.metrics import (
    BaseMetric,
)
from deepeval.models.retry_policy import (
    set_outer_deadline,
    reset_outer_deadline,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
)
from deepeval.test_run import (
    global_test_run_manager,
    TestRunManager,
)
from deepeval.evaluate.types import TestResult
from deepeval.tracing.types import TraceSpanStatus
from deepeval.config.settings import get_settings
from deepeval.test_run import TEMP_FILE_PATH
from deepeval.confident.api import is_confident
from deepeval.test_run.hyperparameters import (
    process_hyperparameters,
    process_prompts,
)

logger = logging.getLogger(__name__)


def _timeout_msg(action: str, seconds: float) -> str:
    if are_timeouts_disabled():
        return (
            f"Timeout occurred while {action} "
            "(DeepEval timeouts are disabled; this likely came from the model/provider SDK or network layer). "
            "Set DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
        )
    return (
        f"Timed out after {seconds:.2f}s while {action}. "
        "Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set "
        "DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
    )


def _log_gather_timeout(
    logger,
    *,
    exc: Optional[BaseException] = None,
    pending: Optional[int] = None,
) -> None:
    settings = get_settings()
    if are_timeouts_disabled():
        logger.warning(
            "A task raised %s while waiting for gathered results; DeepEval gather/per-task timeouts are disabled%s. "
            "This likely came from the model/provider SDK or network layer.",
            type(exc).__name__ if exc else "TimeoutError",
            f" (pending={pending})" if pending is not None else "",
            exc_info=settings.DEEPEVAL_LOG_STACK_TRACES,
        )
    else:
        if pending is not None:
            logger.warning(
                "Gather TIMEOUT after %.1fs; pending=%d tasks. "
                "Some metrics may be marked as timed out. "
                "To give tasks more time, consider increasing "
                "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or "
                "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE.",
                get_gather_timeout_seconds(),
                pending,
            )

        else:
            logger.warning(
                "gather TIMEOUT after %.1fs. Some metrics may be marked as timed out. "
                "To give tasks more time, consider increasing "
                "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or "
                "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE.",
                get_gather_timeout_seconds(),
            )


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

        if get_settings().DEEPEVAL_DISABLE_TIMEOUTS:
            return await coro

        return await asyncio.wait_for(coro, timeout=timeout)
    finally:
        reset_outer_deadline(token)


def _execute_metric(
    metric: BaseMetric,
    test_case: Union[LLMTestCase, ConversationalTestCase],
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
    prompt_hash = prompt.hash if is_confident() else None
    key = f"{prompt.alias}_{prompt_hash}"
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
        existing_prompt_keys = {f"{p.alias}_{p.hash}" for p in test_run.prompts}
        new_prompts = process_prompts(span_hyperparameters)
        for new_prompt in new_prompts:
            new_prompt_key = f"{new_prompt.alias}_{new_prompt.hash}"
            if new_prompt_key not in existing_prompt_keys:
                test_run.prompts.append(new_prompt)

    global_test_run_manager.save_test_run(TEMP_FILE_PATH)
