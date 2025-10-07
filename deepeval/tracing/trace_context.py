from typing import Optional
from .context import current_trace_context
from .tracing import trace_manager

class TraceContext:

    def __init__(self):

        current_trace = current_trace_context.get()
        if not current_trace:
            _current_trace = trace_manager.start_new_trace()
            current_trace_context.set(_current_trace)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get_uuid(self):
        return current_trace_context.get().uuid
