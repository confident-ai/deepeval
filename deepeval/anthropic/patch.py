from typing import Callable, List
from functools import wraps

from deepeval.anthropic.extractors import (
    extract_output_parameters,
    extract_input_parameters,
    InputParameters,
    OutputParameters,
)
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import (
    current_trace_context,
    update_current_span,
    update_llm_span,
)
from deepeval.tracing import observe
from deepeval.tracing.trace_context import current_llm_context

_ORIGINAL_METHODS = {}
_ANTHROPIC_PATCHED = False


def patch_anthropic_classes():
    """
    Monkey patch Anthropic resource classes directly.
    """
    global _ANTHROPIC_PATCHED

    # Single guard - if already patched, return immediately
    if _ANTHROPIC_PATCHED:
        return

    try:
        from anthropic.resources.messages import Messages, AsyncMessages

         # Store original methods before patching
        if hasattr(Messages, "create"):
            _ORIGINAL_METHODS["Messages.create"] = Messages.create
            Messages.create = _create_sync_wrapper(Messages.create)

        if hasattr(AsyncMessages, "create"):
            _ORIGINAL_METHODS["AsyncMessages.create"] = AsyncMessages.create
            AsyncMessages.create = _create_async_wrapper(AsyncMessages.create)

    except ImportError:
        pass

    _ANTHROPIC_PATCHED = True

def _create_sync_wrapper(original_method):
    """
    Create a wrapper for sync methods - called ONCE during patching.
    """
    @wraps(original_method)
    def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        patched = _patch_sync_anthropic_client_method(
            orig_method=bound_method
        )
        return patched(*args, **kwargs)

    return method_wrapper

def _create_async_wrapper():
    ...

def _patch_sync_anthropic_client_method():
    ...