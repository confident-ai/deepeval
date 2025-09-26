def get_active_trace_and_span():
    # helper to peek at current trace/span via the observer context
    from deepeval.tracing.context import (
        current_trace_context,
        current_span_context,
    )

    return current_trace_context.get(), current_span_context.get()
