from typing import Callable
from functools import wraps

from deepeval.openai.extractors import (
    extract_output_parameters,
    extract_input_parameters,
    InputParameters,
    OutputParameters,
)
from deepeval.tracing.context import current_trace_context, update_current_span, update_llm_span
from deepeval.tracing import observe
from deepeval.tracing.trace_context import current_prompt_context

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

        @observe(type="llm", model=input_parameters.model)
        async def llm_generation(*args, **kwargs):
            response = await orig_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            update_current_span(
                input=input_parameters.input
                or input_parameters.messages
                or "NA",
                output=output_parameters.output or "NA",
            )
            prompt = current_prompt_context.get()
            update_llm_span(
                input_token_count=output_parameters.prompt_tokens,
                output_token_count=output_parameters.completion_tokens,
                prompt=prompt,
            )
            
            _update_input_and_output_of_current_trace(input_parameters, output_parameters)
            
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

        @observe(type="llm", model=input_parameters.model)
        def llm_generation(*args, **kwargs):
            response = orig_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            update_current_span(
                input=input_parameters.input
                or input_parameters.messages
                or "NA",
                output=output_parameters.output or "NA",
            )

            prompt = current_prompt_context.get()
            update_llm_span(
                input_token_count=output_parameters.prompt_tokens,
                output_token_count=output_parameters.completion_tokens,
                prompt=prompt,
            )
            _update_input_and_output_of_current_trace(input_parameters, output_parameters)  
            
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

def _update_input_and_output_of_current_trace(input_parameters: InputParameters, output_parameters: OutputParameters):
    
    current_trace = current_trace_context.get()
    if current_trace:
        if current_trace.input is None:
            current_trace.input = input_parameters.input or input_parameters.messages or "NA"
        
        current_trace.output = output_parameters.output or "NA"
    return