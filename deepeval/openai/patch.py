from typing import Callable, List
from functools import wraps

from deepeval.openai.extractors import (
    extract_output_parameters,
    extract_input_parameters,
    InputParameters,
    OutputParameters,
)
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import current_trace_context, update_current_span, update_llm_span
from deepeval.tracing import observe
from deepeval.tracing.trace_context import current_llm_context
from deepeval.openai.utils import create_child_tool_spans

def patch_async_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
):
    @wraps(orig_method)
    async def patched_async_openai_method(
        *args,
        **kwargs
    ):
        input_parameters: InputParameters = extract_input_parameters(
            is_completion_method, kwargs
        )

        llm_context = current_llm_context.get()

        @observe(type="llm", model=input_parameters.model, metrics=llm_context.metrics, metric_collection=llm_context.metric_collection)
        async def llm_generation(*args, **kwargs):
            response = await orig_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            _update_all_attributes(
                input_parameters, output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
            )
            
            return response

        return await llm_generation(*args, **kwargs)

    return patched_async_openai_method


def patch_sync_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
):
    @wraps(orig_method)
    def patched_sync_openai_method(
        *args,
        **kwargs
    ):
        input_parameters: InputParameters = extract_input_parameters(
            is_completion_method, kwargs
        )

        llm_context = current_llm_context.get()

        @observe(type="llm", model=input_parameters.model, metrics=llm_context.metrics, metric_collection=llm_context.metric_collection)
        def llm_generation(*args, **kwargs):
            response = orig_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            _update_all_attributes(
                input_parameters, output_parameters,
                llm_context.expected_tools,
                llm_context.expected_output,
                llm_context.context,
                llm_context.retrieval_context,
            )
            
            return response

        return llm_generation(*args, **kwargs)

    return patched_sync_openai_method


def patch_openai_classes():
    """Monkey patch OpenAI resource classes directly."""
    
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
        
        # Helper to create bound method wrapper
        def wrap_sync_method(original_method):
            def method_wrapper(self, *args, **kwargs):
                # Bind the original method to self, then wrap it
                bound_method = original_method.__get__(self, type(self))
                patched = patch_sync_openai_client_method(
                    orig_method=bound_method,
                    is_completion_method=True
                )
                return patched(*args, **kwargs)
            return method_wrapper
        
        def wrap_async_method(original_method):
            async def method_wrapper(self, *args, **kwargs):
                # Bind the original method to self, then wrap it
                bound_method = original_method.__get__(self, type(self))
                patched = patch_async_openai_client_method(
                    orig_method=bound_method,
                    is_completion_method=True
                )
                return await patched(*args, **kwargs)
            return method_wrapper
        
        # Patch sync methods
        if hasattr(Completions, 'create'):
            Completions.create = wrap_sync_method(Completions.create)
        if hasattr(Completions, 'parse'):
            Completions.parse = wrap_sync_method(Completions.parse)
        
        # Patch async methods
        if hasattr(AsyncCompletions, 'create'):
            AsyncCompletions.create = wrap_async_method(AsyncCompletions.create)
        if hasattr(AsyncCompletions, 'parse'):
            AsyncCompletions.parse = wrap_async_method(AsyncCompletions.parse)
            
    except ImportError:
        pass
    
    # Patch responses.create
    try:
        from openai.resources.responses import Responses, AsyncResponses
        
        # Use the same wrapper functions defined above
        def wrap_sync_method(original_method):
            def method_wrapper(self, *args, **kwargs):
                bound_method = original_method.__get__(self, type(self))
                patched = patch_sync_openai_client_method(
                    orig_method=bound_method,
                    is_completion_method=False  # responses use different parameters
                )
                return patched(*args, **kwargs)
            return method_wrapper
        
        def wrap_async_method(original_method):
            async def method_wrapper(self, *args, **kwargs):
                bound_method = original_method.__get__(self, type(self))
                patched = patch_async_openai_client_method(
                    orig_method=bound_method,
                    is_completion_method=False  # responses use different parameters
                )
                return await patched(*args, **kwargs)
            return method_wrapper
        
        # Patch sync and async responses.create
        if hasattr(Responses, 'create'):
            Responses.create = wrap_sync_method(Responses.create)
        if hasattr(AsyncResponses, 'create'):
            AsyncResponses.create = wrap_async_method(AsyncResponses.create)
            
    except ImportError:
        pass

def _update_all_attributes(
    input_parameters: InputParameters,
    output_parameters: OutputParameters, 
    expected_tools: List[ToolCall],
    expected_output: str,
    context: List[str],
    retrieval_context: List[str],
):
    """Update span and trace attributes with input/output parameters."""
    update_current_span(
        input=input_parameters.input
        or input_parameters.messages
        or "NA",
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
        input_token_count=output_parameters.prompt_tokens,
        output_token_count=output_parameters.completion_tokens,
        prompt=llm_context.prompt,
    )

    if output_parameters.tools_called:
        create_child_tool_spans(output_parameters)

    __update_input_and_output_of_current_trace(input_parameters, output_parameters)

    
def __update_input_and_output_of_current_trace(input_parameters: InputParameters, output_parameters: OutputParameters):
    
    current_trace = current_trace_context.get()
    if current_trace:
        if current_trace.input is None:
            current_trace.input = input_parameters.input or input_parameters.messages
        
        if current_trace.output is None:
            current_trace.output = output_parameters.output

    return