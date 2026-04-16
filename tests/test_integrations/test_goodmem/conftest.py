import pytest

from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.context import current_trace_context, current_span_context


@pytest.fixture(scope="function", autouse=True)
def reset_trace_state():
    """Reset tracing state before each test."""
    trace_manager.clear_traces()
    current_trace_context.set(None)
    current_span_context.set(None)
    yield
