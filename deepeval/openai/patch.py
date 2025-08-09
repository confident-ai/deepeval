from typing import Callable, List, Optional
from functools import wraps

from deepeval.openai.utils import (
    get_attr_path,
    set_attr_path,
    add_test_case,
    create_child_tool_spans,
)
from deepeval.openai.extractors import (
    extract_output_parameters,
    extract_input_parameters,
    InputParameters,
    ToolCall,
)
from deepeval.tracing.context import update_current_span, update_llm_span
from deepeval.tracing import trace_manager, observe
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase


def patch_openai(openai_module):
    if getattr(openai_module, "_deepeval_patched", False):
        return

    openai_module._deepeval_patched = True
    openai_class = getattr(openai_module, "OpenAI", None)
    async_openai_class = getattr(openai_module, "AsyncOpenAI", None)

    if openai_class:
        patch_openai_client(openai_class, is_async=False)
    if async_openai_class:
        patch_openai_client(async_openai_class, is_async=True)


def patch_openai_client(openai_class, is_async: bool):
    original_init = openai_class.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        method_paths = {
            # path → is_completion_method
            "chat.completions.create": True,
            "beta.chat.completions.parse": True,
            "responses.create": False,
        }
        for path, is_completion in method_paths.items():
            method = get_attr_path(self, path)
            if not callable(method):
                continue
            if is_async:
                patched_method = patch_async_openai_client_method(
                    orig_method=method,
                    is_completion_method=is_completion,
                )
            else:
                patched_method = patch_sync_openai_client_method(
                    orig_method=method,
                    is_completion_method=is_completion,
                )
            set_attr_path(self, path, patched_method)

    openai_class.__init__ = new_init


def patch_async_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
):
    @wraps(orig_method)
    async def patched_async_openai_method(
        metrics: Optional[List[BaseMetric]] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
        expected_tools: Optional[List[ToolCall]] = None,
        *args,
        **kwargs
    ):
        input_parameters: InputParameters = extract_input_parameters(
            is_completion_method, kwargs
        )
        is_traced = len(trace_manager.traces) > 0

        if is_traced:

            @observe(type="llm", model=input_parameters.model, metrics=metrics)
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
                    expected_output=expected_output,
                    retrieval_context=retrieval_context,
                    context=context,
                    tools_called=output_parameters.tools_called,
                    expected_tools=expected_tools,
                )
                update_llm_span(
                    input_token_count=output_parameters.prompt_tokens,
                    output_token_count=output_parameters.completion_tokens,
                )
                create_child_tool_spans(output_parameters)
                return response

            return await llm_generation(*args, **kwargs)
        else:
            response = await orig_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            test_case = LLMTestCase(
                input=input_parameters.input,
                actual_output=output_parameters.output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
                context=context,
                tools_called=output_parameters.tools_called,
                expected_tools=expected_tools,
            )
            add_test_case(
                test_case=test_case,
                metrics=metrics,
                input_parameters=input_parameters,
            )
            return response

    return patched_async_openai_method


def patch_sync_openai_client_method(
    orig_method: Callable,
    is_completion_method: bool = False,
):
    @wraps(orig_method)
    def patched_sync_openai_method(
        metrics: Optional[List[BaseMetric]] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
        expected_tools: Optional[List[ToolCall]] = None,
        *args,
        **kwargs
    ):
        input_parameters: InputParameters = extract_input_parameters(
            is_completion_method, kwargs
        )
        is_traced = len(trace_manager.traces) > 0

        if is_traced:

            @observe(type="llm", model=input_parameters.model, metrics=metrics)
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
                    expected_output=expected_output,
                    retrieval_context=retrieval_context,
                    context=context,
                    tools_called=output_parameters.tools_called,
                    expected_tools=expected_tools,
                )
                update_llm_span(
                    input_token_count=output_parameters.prompt_tokens,
                    output_token_count=output_parameters.completion_tokens,
                )
                create_child_tool_spans(output_parameters)
                return response

            return llm_generation(*args, **kwargs)
        else:
            response = orig_method(*args, **kwargs)
            output_parameters = extract_output_parameters(
                is_completion_method, response, input_parameters
            )
            test_case = LLMTestCase(
                input=input_parameters.input,
                actual_output=output_parameters.output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
                context=context,
                tools_called=output_parameters.tools_called,
                expected_tools=expected_tools,
            )
            add_test_case(
                test_case=test_case,
                metrics=metrics,
                input_parameters=input_parameters,
            )
            return response

    return patched_sync_openai_method
