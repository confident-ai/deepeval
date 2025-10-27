from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from deepeval.metrics import BaseMetric
from deepeval.prompt import Prompt
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import current_trace_context, update_current_trace
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import TraceWorkerStatus
from deepeval.tracing.utils import is_async_context


@dataclass
class LlmSpanContext:
    prompt: Optional[Prompt] = None
    metrics: Optional[List[BaseMetric]] = None
    metric_collection: Optional[str] = None
    expected_output: Optional[str] = None
    expected_tools: Optional[List[ToolCall]] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None


@dataclass
class AgentSpanContext:
    metrics: Optional[List[BaseMetric]] = None
    metric_collection: Optional[str] = None
    expected_output: Optional[str] = None
    expected_tools: Optional[List[ToolCall]] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None


current_llm_context: ContextVar[Optional[LlmSpanContext]] = ContextVar(
    "current_llm_context", default=LlmSpanContext()
)

current_agent_context: ContextVar[Optional[AgentSpanContext]] = ContextVar(
    "current_agent_context", default=AgentSpanContext()
)


@contextmanager
def trace(
    llm_span_context: Optional[LlmSpanContext] = None,
    agent_span_context: Optional[AgentSpanContext] = None,
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
    metrics: Optional[List[BaseMetric]] = None,
    metric_collection: Optional[str] = None,
):
    if is_async_context():
        trace_manager._print_trace_status(
            message="Warning: Detected use of the synchronous 'trace' context manager within an async method",
            trace_worker_status=TraceWorkerStatus.WARNING,
            description="Wrapping an async method with the synchronous 'trace' context manager may lead to unexpected behavior.",
        )

    current_trace = current_trace_context.get()

    if not current_trace:
        current_trace = trace_manager.start_new_trace()

    if metrics:
        current_trace.metrics = metrics

    if metric_collection:
        current_trace.metric_collection = metric_collection

    current_trace_context.set(current_trace)

    update_current_trace(
        name=name,
        tags=tags,
        metadata=metadata,
        thread_id=thread_id,
        user_id=user_id,
        input=input,
        output=output,
        retrieval_context=retrieval_context,
        context=context,
        expected_output=expected_output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )

    if llm_span_context:
        current_llm_context.set(llm_span_context)
    if agent_span_context:
        current_agent_context.set(agent_span_context)
    try:
        yield current_trace
    finally:
        current_llm_context.set(LlmSpanContext())
        current_agent_context.set(AgentSpanContext())
