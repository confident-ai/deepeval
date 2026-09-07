from functools import wraps
from importlib import import_module
from typing import Any, Callable, Dict

from deepeval.model_integrations.gateways import OPENROUTER
from deepeval.model_integrations.utils import _update_all_attributes
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

# Every resource class exposes the same sync/async pair.
_PATCH_TARGETS = (
    ("openrouter.chat", "Chat"),
    ("openrouter.responses", "Responses"),
)
_PATCHED_METHODS = ("send", "send_async")


def _resource_classes():
    """Yield the (name, class) pairs present in the installed SDK."""
    for module_path, class_name in _PATCH_TARGETS:
        try:
            module = import_module(module_path)
        except ImportError:
            continue
        resource = getattr(module, class_name, None)
        if resource is not None:
            yield class_name, resource


def patch_openrouter_classes():
    """Monkey patch OpenRouter resource classes directly."""
    global _OPENROUTER_PATCHED

    if _OPENROUTER_PATCHED:
        return

    for class_name, resource in _resource_classes():
        for method_name in _PATCHED_METHODS:
            method = getattr(resource, method_name, None)
            if method is None:
                continue
            key = f"{class_name}.{method_name}"
            _ORIGINAL_METHODS[key] = method
            wrap = (
                _create_async_wrapper
                if method_name.endswith("_async")
                else _create_sync_wrapper
            )
            setattr(resource, method_name, wrap(method))

    _OPENROUTER_PATCHED = True


def unpatch_openrouter_classes():
    """Restore OpenRouter resource classes to their original state."""
    global _OPENROUTER_PATCHED

    if not _OPENROUTER_PATCHED:
        return

    for class_name, resource in _resource_classes():
        for method_name in _PATCHED_METHODS:
            original = _ORIGINAL_METHODS.get(f"{class_name}.{method_name}")
            if original is not None:
                setattr(resource, method_name, original)

    _OPENROUTER_PATCHED = False


def _is_streaming(kwargs: Dict[str, Any]) -> bool:
    """Streamed responses have nothing to extract until they're consumed."""
    return bool(kwargs.get("stream"))


def _create_sync_wrapper(original_method):
    @wraps(original_method)
    def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        if _is_streaming(kwargs):
            return bound_method(*args, **kwargs)
        patched = _patch_sync_openrouter_client_method(bound_method)
        return patched(*args, **kwargs)

    return method_wrapper


def _create_async_wrapper(original_method):
    @wraps(original_method)
    async def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        if _is_streaming(kwargs):
            return await bound_method(*args, **kwargs)
        patched = _patch_async_openrouter_client_method(bound_method)
        return await patched(*args, **kwargs)

    return method_wrapper


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
                llm_context.provider or Provider.OPEN_ROUTER.value,
                metadata_key=OPENROUTER.metadata_key,
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
                llm_context.provider or Provider.OPEN_ROUTER.value,
                metadata_key=OPENROUTER.metadata_key,
            )
            return response

        return await llm_generation(*args, **kwargs)

    return patched_async_openrouter_method
