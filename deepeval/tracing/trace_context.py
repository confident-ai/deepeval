from .context import current_trace_context
from .tracing import trace_manager
from contextlib import contextmanager


@contextmanager
def trace():
    current_trace = current_trace_context.get()

    if not current_trace:
        current_trace = trace_manager.start_new_trace()
        current_trace_context.set(current_trace)

    yield current_trace
