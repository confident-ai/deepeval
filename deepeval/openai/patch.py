from typing import Callable, List, Optional
from functools import wraps
import inspect
import uuid

from deepeval.openai.evaluate import log_hyperparameters, add_test_case
from deepeval.tracing.attributes import LlmAttributes, ToolAttributes
from deepeval.openai.utils import get_attr_path, set_attr_path
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import trace_manager, observe
from deepeval.metrics.base_metric import BaseMetric

from deepeval.tracing.types import (
    TraceSpanStatus,
    LlmAttributes,
    ToolSpan
)
from deepeval.tracing.context import (
    current_span_context,
    update_current_span
)
from deepeval.openai.extractors import (
    extract_output_parameters,
    extract_input_parameters,
    OutputParameters,
    InputParameters
)

def patch_openai(openai_module):
    if getattr(openai_module, "_deepeval_patched", False):
        return
    openai_module._deepeval_patched = True

    def wrap_init(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            wrap_openai_client_methods(self)

        cls.__init__ = new_init

    for cls_name in ("OpenAI", "AsyncOpenAI"):
        cls = getattr(openai_module, cls_name, None)
        if isinstance(cls, type):
            wrap_init(cls)


##############################################
# Wrap methods in OpenAI Client
##############################################

def wrap_openai_client_methods(client):
    method_paths = {
        # path → is_completion_method
        "chat.completions.create": True,
        "beta.chat.completions.parse": True,
        "responses.create": False,
    }
    for path, is_completion in method_paths.items():
        method = get_attr_path(client, path)
        if callable(method):
            patched_method = generate_patched_openai_method(method, is_completion_method=is_completion)
            set_attr_path(client, path, patched_method)
        

def generate_patched_openai_method(
    orig_method: Callable,
    is_completion_method: bool = False,
):
    is_async = inspect.iscoroutinefunction(orig_method)

    if is_async:
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
            input_parameters: InputParameters = extract_input_parameters(is_completion_method, kwargs)
            is_observed = len(trace_manager.traces) > 0

            if is_observed:
                @observe(type="llm", model=input_parameters.model)
                async def llm_generation(*args, **kwargs):
                    response = await orig_method(*args, **kwargs)
                    output_parameters = extract_output_parameters(is_completion_method, response, input_parameters)
                    update_current_span(
                        test_case=LLMTestCase(
                            input=input_parameters.input,
                            actual_output=output_parameters.output,
                            expected_output=expected_output,
                            retrieval_context=retrieval_context,
                            context=context,
                            tools_called=output_parameters.tools_called,
                            expected_tools=expected_tools
                        ),
                        attributes=LlmAttributes(
                            input=input_parameters.input or input_parameters.messages or "NA",
                            output=output_parameters.output or "NA",
                            input_token_count=output_parameters.prompt_tokens,
                            output_token_count=output_parameters.completion_tokens,
                        )
                    )
                    create_child_tool_spans(output_parameters)
                    return response

                return await llm_generation(*args, **kwargs)
            else:
                response = await orig_method(*args, **kwargs)
                log_hyperparameters(input_parameters)
                output_parameters = extract_output_parameters(is_completion_method, response, input_parameters)
                test_case = LLMTestCase(
                    input=input_parameters.input,
                    actual_output=output_parameters.output,
                    expected_output=expected_output,
                    retrieval_context=retrieval_context,
                    context=context,
                    tools_called=output_parameters.tools_called,
                    expected_tools=expected_tools
                )
                add_test_case(test_case=test_case, metrics=metrics)
                return response

        return patched_async_openai_method

    else:
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
            input_parameters: InputParameters = extract_input_parameters(is_completion_method, kwargs)
            is_observed = len(trace_manager.traces) > 0

            if is_observed:
                @observe(type="llm", model=input_parameters.model)
                def llm_generation(*args, **kwargs):
                    response = orig_method(*args, **kwargs)
                    output_parameters = extract_output_parameters(is_completion_method, response, input_parameters)
                    update_current_span(
                        test_case=LLMTestCase(
                            input=input_parameters.input,
                            actual_output=output_parameters.output,
                            expected_output=expected_output,
                            retrieval_context=retrieval_context,
                            context=context,
                            tools_called=output_parameters.tools_called,
                            expected_tools=expected_tools
                        ),
                        attributes=LlmAttributes(
                            input=input_parameters.input or input_parameters.messages or "NA",
                            output=output_parameters.output or "NA",
                            input_token_count=output_parameters.prompt_tokens,
                            output_token_count=output_parameters.completion_tokens,
                        )
                    )
                    create_child_tool_spans(output_parameters)
                    return response

                return llm_generation(*args, **kwargs)
            else:
                response = orig_method(*args, **kwargs)
                log_hyperparameters(input_parameters)
                output_parameters = extract_output_parameters(is_completion_method, response, input_parameters)
                test_case = LLMTestCase(
                    input=input_parameters.input,
                    actual_output=output_parameters.output,
                    expected_output=expected_output,
                    retrieval_context=retrieval_context,
                    context=context,
                    tools_called=output_parameters.tools_called,
                    expected_tools=expected_tools
                )
                add_test_case(test_case=test_case, metrics=metrics)
                return response

        return patched_sync_openai_method


##############################################
# Tool-Calling Span
##############################################

def create_child_tool_spans(
    output_parameters: OutputParameters
):
    current_span = current_span_context.get()
    if output_parameters.tools_called is not None:
        for tool_called in output_parameters.tools_called:
            tool_span = ToolSpan(
                **{
                    "uuid": str(uuid.uuid4()),
                    "trace_uuid": current_span.trace_uuid,
                    "parent_uuid": current_span.uuid,
                    "start_time": current_span.start_time,
                    "end_time": current_span.start_time,
                    "status": TraceSpanStatus.SUCCESS,
                    "children": [],
                    "name": tool_called.name,
                    "input": tool_called.input_parameters,
                    "output": None,
                    "metrics": None,
                    "attributes": ToolAttributes(
                        input=tool_called.input_parameters,
                        output=None
                    ),
                    "description": tool_called.description
                }
            )
            current_span.children.append(tool_span)