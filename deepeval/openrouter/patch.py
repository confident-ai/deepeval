from functools import wraps
from typing import Callable

from deepeval.model_integrations.utils import (
    OPENROUTER_METADATA_KEY,
    _update_all_attributes,
)
from deepeval.openrouter.extractors import (
    InputParameters,
    safe_extract_input_parameters,
    safe_extract_output_parameters,
)
from deepeval.tracing import observe
from deepeval.tracing.integrations import Integration, Provider
from deepeval.tracing.trace_context import current_llm_context

_ORIGINAL_METHODS = {}
_OPENROUTER_PATCHED = False


def patch_openrouter_classes():
    """Monkey patch OpenRouter resource classes directly."""
    global _OPENROUTER_PATCHED

    # Single guard - if already patched, return immediately
    if _OPENROUTER_PATCHED:
        return

    try:
        from openrouter.chat import Chat

        if hasattr(Chat, "send"):
            _ORIGINAL_METHODS["Chat.send"] = Chat.send
            Chat.send = _create_sync_wrapper(Chat.send)

        if hasattr(Chat, "send_async"):
            _ORIGINAL_METHODS["Chat.send_async"] = Chat.send_async
            Chat.send_async = _create_async_wrapper(Chat.send_async)

    except ImportError:
        pass

    try:
        from openrouter.responses import Responses

        if hasattr(Responses, "send"):
            _ORIGINAL_METHODS["Responses.send"] = Responses.send
            Responses.send = _create_sync_wrapper(Responses.send)

        if hasattr(Responses, "send_async"):
            _ORIGINAL_METHODS["Responses.send_async"] = Responses.send_async
            Responses.send_async = _create_async_wrapper(Responses.send_async)

    except ImportError:
        pass

    # Set flag at the END after successful patching
    _OPENROUTER_PATCHED = True


def _is_streaming(kwargs) -> bool:
    return bool(kwargs.get("stream"))


def _create_sync_wrapper(original_method):
    """Create a wrapper for sync methods - called ONCE during patching."""

    @wraps(original_method)
    def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        if _is_streaming(kwargs):
            return bound_method(*args, **kwargs)
        patched = _patch_sync_openrouter_client_method(bound_method)
        return patched(*args, **kwargs)

    return method_wrapper


def _create_async_wrapper(original_method):
    """Create a wrapper for async methods - called ONCE during patching."""

    @wraps(original_method)
    async def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        if _is_streaming(kwargs):
            return await bound_method(*args, **kwargs)
        patched = _patch_async_openrouter_client_method(bound_method)
        return await patched(*args, **kwargs)

    return method_wrapper


def _resolve_provider() -> str:
    """OpenRouter unless the user explicitly labelled it otherwise."""
    llm_context = current_llm_context.get()
    if llm_context and llm_context.provider:
        return llm_context.provider
    return Provider.OPEN_ROUTER.value


def _patch_sync_openrouter_client_method(original_method: Callable):
    @wraps(original_method)
    def patched_sync_openrouter_method(*args, **kwargs):
        input_parameters: InputParameters = safe_extract_input_parameters(
            kwargs
        )
        llm_context = current_llm_context.get()

        @observe(
            type="llm",
            model=input_parameters.model,
            metrics=llm_context.metrics,
            metric_collection=llm_context.metric_collection,
        )
        def llm_generation(*args, **kwargs):
            response = original_method(*args, **kwargs)
            output_parameters = safe_extract_output_parameters(
                response, input_parameters
            )
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
                Integration.OPEN_ROUTER.value,
                _resolve_provider(),
                OPENROUTER_METADATA_KEY,
            )
            return response

        return llm_generation(*args, **kwargs)

    return patched_sync_openrouter_method


def _patch_async_openrouter_client_method(original_method: Callable):
    @wraps(original_method)
    async def patched_async_openrouter_method(*args, **kwargs):
        input_parameters: InputParameters = safe_extract_input_parameters(
            kwargs
        )
        llm_context = current_llm_context.get()

        @observe(
            type="llm",
            model=input_parameters.model,
            metrics=llm_context.metrics,
            metric_collection=llm_context.metric_collection,
        )
        async def llm_generation(*args, **kwargs):
            response = await original_method(*args, **kwargs)
            output_parameters = safe_extract_output_parameters(
                response, input_parameters
            )
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
                Integration.OPEN_ROUTER.value,
                _resolve_provider(),
                OPENROUTER_METADATA_KEY,
            )
            return response

        return await llm_generation(*args, **kwargs)

    return patched_async_openrouter_method


def unpatch_openrouter_classes():
    """Restore OpenRouter resource classes to their original state."""
    global _OPENROUTER_PATCHED

    if not _OPENROUTER_PATCHED:
        return

    try:
        from openrouter.chat import Chat

        if "Chat.send" in _ORIGINAL_METHODS:
            Chat.send = _ORIGINAL_METHODS["Chat.send"]

        if "Chat.send_async" in _ORIGINAL_METHODS:
            Chat.send_async = _ORIGINAL_METHODS["Chat.send_async"]

    except ImportError:
        pass

    try:
        from openrouter.responses import Responses

        if "Responses.send" in _ORIGINAL_METHODS:
            Responses.send = _ORIGINAL_METHODS["Responses.send"]

        if "Responses.send_async" in _ORIGINAL_METHODS:
            Responses.send_async = _ORIGINAL_METHODS["Responses.send_async"]

    except ImportError:
        pass

    _OPENROUTER_PATCHED = False
