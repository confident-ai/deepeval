from typing import Any, Dict, List, Optional
from contextvars import ContextVar
from collections.abc import Mapping

from deepeval.utils import is_missing
from deepeval.contextvars import get_current_golden
from deepeval.tracing.types import BaseSpan, Trace
from deepeval.test_case.llm_test_case import ToolCall, LLMTestCase
from deepeval.tracing.types import LlmSpan, RetrieverSpan
from deepeval.prompt.prompt import Prompt

current_span_context: ContextVar[Optional[BaseSpan]] = ContextVar(
    "current_span", default=None
)

current_trace_context: ContextVar[Optional[Trace]] = ContextVar(
    "current_trace", default=None
)


def _normalize(s: Optional[str]) -> Optional[str]:
    return None if s is None else str(s).strip().casefold()


def _pick_candidate_input(explicit: Optional[str]) -> Optional[str]:
    """Choose the best input string to compare against the active Golden.

    Precedence:
      1. The explicit `explicit` argument, if it exists.
      2. The current span's `input`, if it exists.
      3. The current observer's call-args (span._function_kwargs) using the first
         presence of: "input", "query", "prompt", "text".
      4. The current trace's `input`.

    Returns the chosen input or None if none are available.
    """

    if not is_missing(explicit):
        return explicit

    span = current_span_context.get()
    if span:
        si = getattr(span, "input", None)
        if not is_missing(si):
            return si
        fk = getattr(span, "_function_kwargs", None)
        if isinstance(fk, Mapping):
            for key in ("input", "query", "prompt", "text"):
                val = fk.get(key)
                if not is_missing(val):
                    return val

    trace = current_trace_context.get()
    if trace and not is_missing(getattr(trace, "input", None)):
        return trace.input
    return None


def _resolve_expected_output_from_context(
    current_value: Optional[str],
    candidate_input: Optional[str] = None,
) -> Optional[str]:
    """Resolve `expected_output` from the active Golden only when appropriate.

    If `current_value` is present, it is returned unchanged.
    Otherwise, try to read the active Golden's `expected_output`, but only when
    the Golden's `input` (case/space-insensitive) matches `candidate_input`.
    Returns the resolved value or the original `current_value` when no match.
    """
    if not is_missing(current_value):
        return current_value
    golden = get_current_golden()
    if not golden:
        return current_value

    golden_input = getattr(golden, "input", None)
    if golden_input is not None and candidate_input is not None:
        if _normalize(str(golden_input)) != _normalize(str(candidate_input)):
            return current_value
    exp = getattr(golden, "expected_output", None)
    return exp if not is_missing(exp) else current_value


def update_current_span(
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    test_case: Optional[LLMTestCase] = None,
):
    current_span = current_span_context.get()
    if not current_span:
        return
    if test_case:
        # attempt to retrieve expected_output from the active Golden if caller omitted it.
        test_case.expected_output = _resolve_expected_output_from_context(
            test_case.expected_output,
            candidate_input=_pick_candidate_input(test_case.input),
        )

        current_span.input = test_case.input
        current_span.output = test_case.actual_output
        current_span.expected_output = test_case.expected_output
        current_span.retrieval_context = test_case.retrieval_context
        current_span.context = test_case.context
        current_span.tools_called = test_case.tools_called
        current_span.expected_tools = test_case.expected_tools
    if metadata:
        current_span.metadata = metadata
    if input:
        current_span.input = input
    if output:
        current_span.output = output
    if retrieval_context:
        current_span.retrieval_context = retrieval_context
    if context:
        current_span.context = context
    if expected_output:
        current_span.expected_output = expected_output
    elif not getattr(current_span, "expected_output", None):
        # if still missing, attempt to resolve from context
        current_span.expected_output = _resolve_expected_output_from_context(
            None,
            candidate_input=_pick_candidate_input(
                getattr(current_span, "input", None)
            ),
        )
    if tools_called:
        current_span.tools_called = tools_called
    if expected_tools:
        current_span.expected_tools = expected_tools
    if name:
        current_span.name = name


def update_current_trace(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None,
    expected_output: Optional[str] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    test_case: Optional[LLMTestCase] = None,
):
    current_trace = current_trace_context.get()
    if not current_trace:
        return
    if test_case:
        # resolve expected_output for the trace if caller omitted it.
        test_case.expected_output = _resolve_expected_output_from_context(
            test_case.expected_output,
            candidate_input=_pick_candidate_input(test_case.input),
        )

        current_trace.input = test_case.input
        current_trace.output = test_case.actual_output
        current_trace.expected_output = test_case.expected_output
        current_trace.retrieval_context = test_case.retrieval_context
        current_trace.context = test_case.context
        current_trace.tools_called = test_case.tools_called
        current_trace.expected_tools = test_case.expected_tools
    if name:
        current_trace.name = name
    if tags:
        current_trace.tags = tags
    if metadata:
        current_trace.metadata = metadata
    if thread_id:
        current_trace.thread_id = thread_id
    if user_id:
        current_trace.user_id = user_id
    if input:
        current_trace.input = input
    if output:
        current_trace.output = output
    if retrieval_context:
        current_trace.retrieval_context = retrieval_context
    if context:
        current_trace.context = context
    if expected_output:
        current_trace.expected_output = expected_output
    elif not getattr(current_trace, "expected_output", None):
        current_trace.expected_output = _resolve_expected_output_from_context(
            None,
            candidate_input=_pick_candidate_input(
                getattr(current_trace, "input", None)
            ),
        )
    if tools_called:
        current_trace.tools_called = tools_called
    if expected_tools:
        current_trace.expected_tools = expected_tools


def update_llm_span(
    model: Optional[str] = None,
    input_token_count: Optional[float] = None,
    output_token_count: Optional[float] = None,
    cost_per_input_token: Optional[float] = None,
    cost_per_output_token: Optional[float] = None,
    token_intervals: Optional[Dict[float, str]] = None,
    prompt: Optional[Prompt] = None,
):
    current_span = current_span_context.get()
    if not current_span or not isinstance(current_span, LlmSpan):
        return
    if model:
        current_span.model = model
    if input_token_count:
        current_span.input_token_count = input_token_count
    if output_token_count:
        current_span.output_token_count = output_token_count
    if cost_per_input_token:
        current_span.cost_per_input_token = cost_per_input_token
    if cost_per_output_token:
        current_span.cost_per_output_token = cost_per_output_token
    if token_intervals:
        current_span.token_intervals = token_intervals
    if prompt:
        current_span.prompt = prompt


def update_retriever_span(
    embedder: Optional[str] = None,
    top_k: Optional[int] = None,
    chunk_size: Optional[int] = None,
):
    current_span = current_span_context.get()
    if not current_span or not isinstance(current_span, RetrieverSpan):
        return
    if embedder:
        current_span.embedder = embedder
    if top_k:
        current_span.top_k = top_k
    if chunk_size:
        current_span.chunk_size = chunk_size
