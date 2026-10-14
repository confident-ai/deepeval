from typing import Callable, List
from functools import wraps


from deepeval.openai.extractors import (
    safe_extract_output_parameters,
    safe_extract_input_parameters,
    InputParameters,
    OutputParameters,
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
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.openai.stream import AsyncTracedStream, TracedStream

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


def _create_sync_wrapper(original_method, is_completion_method: bool):
    """Create a wrapper for sync methods - called ONCE during patching."""

    @wraps(original_method)
    def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        patched = _patch_sync_openai_client_method(
            orig_method=bound_method, is_completion_method=is_completion_method
        )
        return patched(*args, **kwargs)

    return method_wrapper


def _create_async_wrapper(original_method, is_completion_method: bool):
    """Create a wrapper for async methods - called ONCE during patching."""

    @wraps(original_method)
    async def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        patched = _patch_async_openai_client_method(
            orig_method=bound_method, is_completion_method=is_completion_method
        )
        return await patched(*args, **kwargs)

    return method_wrapper


def _patch_async_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
):
    @wraps(orig_method)
    async def patched_async_openai_method(*args, **kwargs):
        input_parameters: InputParameters = safe_extract_input_parameters(
            is_completion_method, kwargs
        )

        llm_context = current_llm_context.get()

        if kwargs.get("stream"):
            observer = _stream_observer(input_parameters, llm_context)
            parent_span = current_span_context.get()
            parent_trace = current_trace_context.get()
            observer.__enter__()
            span = current_span_context.get()
            trace = current_trace_context.get()
            try:
                response = await orig_method(*args, **kwargs)
                if not hasattr(response, "__anext__"):
                    output_parameters = safe_extract_output_parameters(
                        is_completion_method, response, input_parameters
                    )
                    _update_all_attributes(
                        input_parameters,
                        output_parameters,
                        llm_context.expected_tools,
                        llm_context.expected_output,
                        llm_context.context,
                        llm_context.retrieval_context,
                        llm_context,
                    )
                    observer.__exit__(None, None, None)
                    return response
            except BaseException as exc:
                observer.__exit__(type(exc), exc, exc.__traceback__)
                raise
            finally:
                current_span_context.set(parent_span)
                current_trace_context.set(parent_trace)
            return AsyncTracedStream(
                response,
                observer,
                span,
                trace,
                is_completion_method,
                input_parameters,
                llm_context,
                _update_all_attributes,
            )

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
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
            )

            return response

        return await llm_generation(*args, **kwargs)

    return patched_async_openai_method


def _patch_sync_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
):
    @wraps(orig_method)
    def patched_sync_openai_method(*args, **kwargs):
        input_parameters: InputParameters = safe_extract_input_parameters(
            is_completion_method, kwargs
        )

        llm_context = current_llm_context.get()

        if kwargs.get("stream"):
            observer = _stream_observer(input_parameters, llm_context)
            parent_span = current_span_context.get()
            parent_trace = current_trace_context.get()
            observer.__enter__()
            span = current_span_context.get()
            trace = current_trace_context.get()
            try:
                response = orig_method(*args, **kwargs)
                if not hasattr(response, "__next__"):
                    output_parameters = safe_extract_output_parameters(
                        is_completion_method, response, input_parameters
                    )
                    _update_all_attributes(
                        input_parameters,
                        output_parameters,
                        llm_context.expected_tools,
                        llm_context.expected_output,
                        llm_context.context,
                        llm_context.retrieval_context,
                        llm_context,
                    )
                    observer.__exit__(None, None, None)
                    return response
            except BaseException as exc:
                observer.__exit__(type(exc), exc, exc.__traceback__)
                raise
            finally:
                current_span_context.set(parent_span)
                current_trace_context.set(parent_trace)
            return TracedStream(
                response,
                observer,
                span,
                trace,
                is_completion_method,
                input_parameters,
                llm_context,
                _update_all_attributes,
            )

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
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
            )

            return response

        return llm_generation(*args, **kwargs)

    return patched_sync_openai_method


def _stream_observer(input_parameters, llm_context):
    return Observer(
        "llm",
        func_name="llm_generation",
        metrics=llm_context.metrics,
        metric_collection=llm_context.metric_collection,
        observe_kwargs={"model": input_parameters.model},
    )


def _update_all_attributes(
    input_parameters: InputParameters,
    output_parameters: OutputParameters,
    expected_tools: List[ToolCall],
    expected_output: str,
    context: List[str],
    retrieval_context: List[str],
    llm_context=None,
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

    llm_context = llm_context or current_llm_context.get()

    update_llm_span(
        input_token_count=output_parameters.prompt_tokens,
        output_token_count=output_parameters.completion_tokens,
        prompt=llm_context.prompt,
    )
    current_span = current_span_context.get()
    if isinstance(current_span, LlmSpan):
        current_span.integration = Integration.OPEN_AI.value
        current_span.provider = Provider.OPEN_AI.value
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
