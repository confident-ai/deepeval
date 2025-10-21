from typing import Callable, List
from functools import wraps

from deepeval.anthropic.extractors import (
    extract_input_parameters,
    extract_output_parameters,
    InputParameters,
    OutputParameters,
)
from deepeval.anthropic.utils import create_child_tool_spans
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
            original_method=bound_method
        )
        return patched(*args, **kwargs)

    return method_wrapper


def _create_async_wrapper(original_method):
    """
    Create a wrapper for sync methods - called ONCE during patching.
    """

    @wraps(original_method)
    def method_wrapper(self, *args, **kwargs):
        bound_method = original_method.__get__(self, type(self))
        patched = _patch_async_anthropic_client_method(
            original_method=bound_method
        )
        return patched(*args, **kwargs)

    return method_wrapper


def _patch_sync_anthropic_client_method(original_method: Callable):
    @wraps(original_method)
    def patched_sync_anthropic_method(*args, **kwargs):
        input_parameters: InputParameters = extract_input_parameters(kwargs)
        llm_context = current_llm_context.get()

        @observe(
            type="llm",
            model=input_parameters.model,
            metrics=llm_context.metrics,
            metric_collection=llm_context.metric_collection,
        )
        def llm_generation(*args, **kwargs):
            messages_api_response = original_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                messages_api_response, input_parameters
            )
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
            )
            return messages_api_response

        return llm_generation(*args, **kwargs)

    return patched_sync_anthropic_method


def _patch_async_anthropic_client_method(original_method: Callable):
    @wraps(original_method)
    async def patched_async_anthropic_method(*args, **kwargs):
        input_parameters: InputParameters = extract_input_parameters(kwargs)
        llm_context = current_llm_context.get()

        @observe(
            type="llm",
            model=input_parameters.model,
            metrics=llm_context.metrics,
            metric_collection=llm_context.metric_collection,
        )
        async def llm_generation(*args, **kwargs):
            messages_api_response = await original_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                messages_api_response, input_parameters
            )
            _update_all_attributes(
                input_parameters,
                output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
            )
            return messages_api_response

        return await llm_generation(*args, **kwargs)

    return patched_async_anthropic_method


def _update_all_attributes(
    input_parameters: InputParameters,
    output_parameters: OutputParameters,
    expected_tools: List[ToolCall],
    expected_output: str,
    context: List[str],
    retrieval_context: List[str],
):
    """
    Update span and trace attributes with input/output parameters.
    """
    update_current_span(
        input=input_parameters.input or input_parameters.messages or "NA",
        output=output_parameters.output or "NA",
        tools_called=output_parameters.tools_called,
        # attributes to be added
        expected_output=expected_output,
        expected_tools=expected_tools,
        context=context,
        retrieval_context=retrieval_context,
    )
    llm_context = current_llm_context.get()
    update_llm_span(
        input_token_count=output_parameters.usage["input_tokens"],
        output_token_count=output_parameters.usage["output_tokens"],
        prompt=llm_context.prompt,
    )
    if output_parameters.tools_called:
        create_child_tool_spans(output_parameters)

    __update_input_and_output_of_current_trace(
        input_parameters, output_parameters
    )


def __update_input_and_output_of_current_trace(
    input_parameters: InputParameters, output_parameters: OutputParameters
):
    current_trace = current_trace_context.get()
    if current_trace:
        if current_trace.input is None:
            current_trace.input = input_parameters.input
        if current_trace.output is None:
            current_trace.output = output_parameters.output

    return


def unpatch_anthropic_classes():
    """
    Restore Anthropic resource classes to their original state.
    """
    global _ANTHROPIC_PATCHED

    # If not patched, nothing to do
    if not _ANTHROPIC_PATCHED:
        return

    try:
        from anthropic.resources.messages import Messages, AsyncMessages

        # Restore original methods for Messages
        if hasattr(Messages, "create"):
            Messages.create = _ORIGINAL_METHODS["Messages.create"]

        if hasattr(AsyncMessages, "create"):
            AsyncMessages.create = _ORIGINAL_METHODS["AsyncMessages.create"]

    except ImportError:
        pass

    # Reset the patched flag
    _ANTHROPIC_PATCHED = False
