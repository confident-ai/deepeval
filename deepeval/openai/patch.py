from typing import Callable, List, Optional, Tuple
from functools import wraps


from deepeval.openai.extractors import (
    safe_extract_output_parameters,
    safe_extract_input_parameters,
    InputParameters,
    OutputParameters,
)
from deepeval.model_integrations.utils import (
    OPENROUTER_METADATA_KEY,
    detect_provider_from_base_url,
    extract_openrouter_metadata,
)
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    update_current_span,
    update_llm_span,
)
from deepeval.tracing import observe
from deepeval.tracing.trace_context import current_llm_context
from deepeval.tracing.types import LlmSpan
from deepeval.tracing.integrations import Integration, Provider
from deepeval.tracing.tracing import trace_manager

# Store original methods for safety and potential unpatching
_ORIGINAL_METHODS = {}
_OPENAI_PATCHED = False


def patch_openai_classes():
    """Monkey patch OpenAI resource classes directly."""
    global _OPENAI_PATCHED

    # Single guard - if already patched, return immediately
    if _OPENAI_PATCHED:
        return

    try:
        from openai.resources.chat.completions import (
            Completions,
            AsyncCompletions,
        )

        # Store original methods before patching
        if hasattr(Completions, "create"):
            _ORIGINAL_METHODS["Completions.create"] = Completions.create
            Completions.create = _create_sync_wrapper(
                Completions.create, is_completion_method=True
            )

        if hasattr(Completions, "parse"):
            _ORIGINAL_METHODS["Completions.parse"] = Completions.parse
            Completions.parse = _create_sync_wrapper(
                Completions.parse, is_completion_method=True
            )

        if hasattr(AsyncCompletions, "create"):
            _ORIGINAL_METHODS["AsyncCompletions.create"] = (
                AsyncCompletions.create
            )
            AsyncCompletions.create = _create_async_wrapper(
                AsyncCompletions.create, is_completion_method=True
            )

        if hasattr(AsyncCompletions, "parse"):
            _ORIGINAL_METHODS["AsyncCompletions.parse"] = AsyncCompletions.parse
            AsyncCompletions.parse = _create_async_wrapper(
                AsyncCompletions.parse, is_completion_method=True
            )

    except ImportError:
        pass

    try:
        from openai.resources.responses import Responses, AsyncResponses

        if hasattr(Responses, "create"):
            _ORIGINAL_METHODS["Responses.create"] = Responses.create
            Responses.create = _create_sync_wrapper(
                Responses.create, is_completion_method=False
            )

        if hasattr(AsyncResponses, "create"):
            _ORIGINAL_METHODS["AsyncResponses.create"] = AsyncResponses.create
            AsyncResponses.create = _create_async_wrapper(
                AsyncResponses.create, is_completion_method=False
            )

    except ImportError:
        pass

    # Set flag at the END after successful patching
    _OPENAI_PATCHED = True


def _resolve_provider(resource) -> Tuple[str, Optional[str]]:
    base_url = getattr(getattr(resource, "_client", None), "base_url", None)
    detected = detect_provider_from_base_url(base_url)

    llm_context = current_llm_context.get()
    override = llm_context.provider if llm_context else None

    return (override or detected or Provider.OPEN_AI.value), detected


def _create_sync_wrapper(original_method, is_completion_method: bool):
    """Create a wrapper for sync methods - called ONCE during patching."""

    @wraps(original_method)
    def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        provider, detected = _resolve_provider(self)
        patched = _patch_sync_openai_client_method(
            orig_method=bound_method,
            is_completion_method=is_completion_method,
            provider=provider,
            detected_provider=detected,
        )
        return patched(*args, **kwargs)

    return method_wrapper


def _create_async_wrapper(original_method, is_completion_method: bool):
    """Create a wrapper for async methods - called ONCE during patching."""

    @wraps(original_method)
    async def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        provider, detected = _resolve_provider(self)
        patched = _patch_async_openai_client_method(
            orig_method=bound_method,
            is_completion_method=is_completion_method,
            provider=provider,
            detected_provider=detected,
        )
        return await patched(*args, **kwargs)

    return method_wrapper


def _patch_async_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
    provider: str = Provider.OPEN_AI.value,
    detected_provider: Optional[str] = None,
):
    @wraps(orig_method)
    async def patched_async_openai_method(*args, **kwargs):
        input_parameters: InputParameters = safe_extract_input_parameters(
            is_completion_method, kwargs
        )

        llm_context = current_llm_context.get()

        @observe(
            type="llm",
            model=input_parameters.model,
            metrics=llm_context.metrics,
            metric_collection=llm_context.metric_collection,
        )
        async def llm_generation(*args, **kwargs):
            response = await orig_method(*args, **kwargs)
            output_parameters = safe_extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            if detected_provider == Provider.OPEN_ROUTER.value:
                output_parameters.metadata = extract_openrouter_metadata(
                    response
                )
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
                provider,
            )

            return response

        return await llm_generation(*args, **kwargs)

    return patched_async_openai_method


def _patch_sync_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
    provider: str = Provider.OPEN_AI.value,
    detected_provider: Optional[str] = None,
):
    @wraps(orig_method)
    def patched_sync_openai_method(*args, **kwargs):
        input_parameters: InputParameters = safe_extract_input_parameters(
            is_completion_method, kwargs
        )

        llm_context = current_llm_context.get()

        @observe(
            type="llm",
            model=input_parameters.model,
            metrics=llm_context.metrics,
            metric_collection=llm_context.metric_collection,
        )
        def llm_generation(*args, **kwargs):
            response = orig_method(*args, **kwargs)
            output_parameters = safe_extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            if detected_provider == Provider.OPEN_ROUTER.value:
                output_parameters.metadata = extract_openrouter_metadata(
                    response
                )
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
                provider,
            )

            return response

        return llm_generation(*args, **kwargs)

    return patched_sync_openai_method


def _update_all_attributes(
    input_parameters: InputParameters,
    output_parameters: OutputParameters,
    expected_tools: List[ToolCall],
    expected_output: str,
    context: List[str],
    retrieval_context: List[str],
    provider: str = Provider.OPEN_AI.value,
):
    """Update span and trace attributes with input/output parameters."""
    update_current_span(
        input=input_parameters.messages,
        output=output_parameters.output or output_parameters.tools_called,
        tools_called=output_parameters.tools_called,
        # attributes to be added
        expected_output=expected_output,
        expected_tools=expected_tools,
        context=context,
        retrieval_context=retrieval_context,
    )

    llm_context = current_llm_context.get()

    update_llm_span(
        input_token_count=output_parameters.prompt_tokens,
        output_token_count=output_parameters.completion_tokens,
        prompt=llm_context.prompt,
    )
    current_span = current_span_context.get()
    if isinstance(current_span, LlmSpan):
        # The integration is the SDK we instrumented; the provider is whoever
        # served the request. They differ whenever the OpenAI SDK is pointed at
        # a gateway.
        current_span.integration = Integration.OPEN_AI.value
        current_span.provider = provider
        if output_parameters.metadata:
            # Merge rather than assign — `update_current_span(metadata=...)`
            # replaces the dict wholesale, so a user who set their own metadata
            # inside this call would otherwise lose it (or lose ours).
            current_span.metadata = {
                **(current_span.metadata or {}),
                OPENROUTER_METADATA_KEY: output_parameters.metadata,
            }
        if current_span.parent_uuid:
            parent_span = trace_manager.get_span_by_uuid(
                current_span.parent_uuid
            )
            if parent_span and not parent_span.integration:
                parent_span.integration = Integration.OPEN_AI.value

    __update_input_and_output_of_current_trace(
        input_parameters, output_parameters
    )


def __update_input_and_output_of_current_trace(
    input_parameters: InputParameters, output_parameters: OutputParameters
):

    current_trace = current_trace_context.get()
    if current_trace:
        if current_trace.input is None:
            current_trace.input = (
                input_parameters.input or input_parameters.messages
            )

        if current_trace.output is None:
            current_trace.output = output_parameters.output

    return


def unpatch_openai_classes():
    """Restore OpenAI resource classes to their original state."""
    global _OPENAI_PATCHED

    # If not patched, nothing to do
    if not _OPENAI_PATCHED:
        return

    try:
        from openai.resources.chat.completions import (
            Completions,
            AsyncCompletions,
        )

        # Restore original methods for Completions
        if "Completions.create" in _ORIGINAL_METHODS:
            Completions.create = _ORIGINAL_METHODS["Completions.create"]

        if "Completions.parse" in _ORIGINAL_METHODS:
            Completions.parse = _ORIGINAL_METHODS["Completions.parse"]

        # Restore original methods for AsyncCompletions
        if "AsyncCompletions.create" in _ORIGINAL_METHODS:
            AsyncCompletions.create = _ORIGINAL_METHODS[
                "AsyncCompletions.create"
            ]

        if "AsyncCompletions.parse" in _ORIGINAL_METHODS:
            AsyncCompletions.parse = _ORIGINAL_METHODS["AsyncCompletions.parse"]

    except ImportError:
        pass

    try:
        from openai.resources.responses import Responses, AsyncResponses

        # Restore original methods for Responses
        if "Responses.create" in _ORIGINAL_METHODS:
            Responses.create = _ORIGINAL_METHODS["Responses.create"]

        # Restore original methods for AsyncResponses
        if "AsyncResponses.create" in _ORIGINAL_METHODS:
            AsyncResponses.create = _ORIGINAL_METHODS["AsyncResponses.create"]

    except ImportError:
        pass

    # Reset the patched flag
    _OPENAI_PATCHED = False
