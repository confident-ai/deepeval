from typing import Any, Dict, List, Optional
from contextvars import ContextVar

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
