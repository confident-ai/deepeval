import pytest
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from deepeval.tracing import observe, trace_manager
from deepeval.tracing.context import current_span_context, current_trace_context


@observe(type="tool")
def child_function():
    return "child result"


@observe(type="agent")
def parent_with_plain_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(child_function)
    future.result()
    executor.shutdown(wait=True)
    return "done"


@observe(type="agent")
def parent_with_copy_context():
    ctx = copy_context()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(ctx.run, child_function)
    future.result()
    executor.shutdown(wait=True)
    return "done"


@pytest.fixture(autouse=True)
def clean_trace_state():
    trace_manager.clear_traces()
    trace_manager.tracing_enabled = False
    current_span_context.set(None)
    current_trace_context.set(None)
    yield
    trace_manager.clear_traces()
    trace_manager.tracing_enabled = True
    current_span_context.set(None)
    current_trace_context.set(None)


def test_threadpool_without_copy_context_creates_two_traces():
    """Without copy_context, the child @observe function in a ThreadPoolExecutor
    creates a separate trace because ContextVar values don't propagate to
    new threads."""
    parent_with_plain_executor()

    assert (
        len(trace_manager.traces) == 2
    ), f"Expected 2 traces (parent + orphaned child), got {len(trace_manager.traces)}"


def test_threadpool_with_copy_context_creates_one_trace():
    """With copy_context, the child @observe function in a ThreadPoolExecutor
    correctly attaches to the parent trace because ContextVar values are
    propagated."""
    parent_with_copy_context()

    assert (
        len(trace_manager.traces) == 1
    ), f"Expected 1 trace (child nested under parent), got {len(trace_manager.traces)}"

    the_trace = trace_manager.traces[0]
    assert len(the_trace.root_spans) == 1, "Expected exactly 1 root span"

    root_span = the_trace.root_spans[0]
    assert (
        len(root_span.children) == 1
    ), "Expected the child function as a child span of the root"
    assert root_span.children[0].name == "child_function"
