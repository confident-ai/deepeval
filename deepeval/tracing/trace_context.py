from typing import Optional
from contextvars import ContextVar
from contextlib import contextmanager

from .tracing import trace_manager
from .context import current_trace_context

from deepeval.prompt import Prompt
current_prompt_context: ContextVar[Optional[Prompt]] = ContextVar(
    "current_prompt", default=None
)

@contextmanager
def trace(prompt: Optional[Prompt] = None):
    current_trace = current_trace_context.get()

    if not current_trace:
        current_trace = trace_manager.start_new_trace()
        current_trace_context.set(current_trace)

    # set the current prompt context
    current_prompt_context.set(prompt)

    yield current_trace
