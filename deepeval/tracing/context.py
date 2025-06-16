from typing import Any, Dict, List, Optional
from contextvars import ContextVar
from deepeval.tracing.types import BaseSpan, Trace, Feedback
from deepeval.test_case import LLMTestCase
from deepeval.tracing.attributes import Attributes


current_span_context: ContextVar[Optional[BaseSpan]] = ContextVar(
    "current_span", default=None
)

current_trace_context: ContextVar[Optional[Trace]] = ContextVar(
    "current_trace", default=None
)


def update_current_span(
    test_case: Optional[LLMTestCase] = None,
    attributes: Optional[Attributes] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feedback: Optional[Feedback] = None,
):
    current_span = current_span_context.get()
    if not current_span:
        return
    if attributes:
        current_span.set_attributes(attributes)
    if test_case:
        current_span.llm_test_case = test_case
    if metadata:
        current_span.metadata = metadata
    if feedback:
        current_span.feedback = feedback


def update_current_trace(
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    feedback: Optional[Feedback] = None,
    test_case: Optional[LLMTestCase] = None,
):
    current_trace = current_trace_context.get()
    if not current_trace:
        return
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
    if feedback:
        current_trace.feedback = feedback
    if test_case:
        current_trace.llm_test_case = test_case
