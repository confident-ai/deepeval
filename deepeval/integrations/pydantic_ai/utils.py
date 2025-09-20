from time import perf_counter
from contextlib import asynccontextmanager
import inspect
import functools
from typing import Any, Callable, List, Optional

from pydantic_ai.models import Model
from pydantic_ai.agent import AgentRunResult
from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from deepeval.prompt import Prompt
from deepeval.tracing.tracing import Observer
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import current_trace_context, current_span_context
from deepeval.tracing.types import AgentSpan, LlmOutput, LlmSpan, LlmToolCall


# llm tools called
def extract_tools_called_from_llm_response(
    result: List[ModelResponsePart],
) -> List[ToolCall]:
    tool_calls = []

    # Loop through each ModelResponsePart
    for part in result:
        # Look for parts with part_kind="tool-call"
        if hasattr(part, "part_kind") and part.part_kind == "tool-call":
            # Extract tool name and args from the ToolCallPart
            tool_name = part.tool_name
            input_parameters = (
                part.args_as_dict() if hasattr(part, "args_as_dict") else None
            )

            # Create and append ToolCall object
            tool_call = ToolCall(
                name=tool_name, input_parameters=input_parameters
            )
            tool_calls.append(tool_call)

    return tool_calls


# TODO: llm tools called (reposne is present next message)
def extract_tools_called(result: AgentRunResult) -> List[ToolCall]:
    tool_calls = []

    # Access the message history from the _state
    message_history = result._state.message_history

    # Scan through all messages in the history
    for message in message_history:
        # Check if this is a ModelResponse (kind="response")
        if hasattr(message, "kind") and message.kind == "response":
            # For ModelResponse messages, check each part
            if hasattr(message, "parts"):
                for part in message.parts:
                    # Look for parts with part_kind="tool-call"
                    if (
                        hasattr(part, "part_kind")
                        and part.part_kind == "tool-call"
                    ):
                        # Extract tool name and args from the ToolCallPart
                        tool_name = part.tool_name
                        input_parameters = (
                            part.args_as_dict()
                            if hasattr(part, "args_as_dict")
                            else None
                        )

                        # Create and append ToolCall object
                        tool_call = ToolCall(
                            name=tool_name, input_parameters=input_parameters
                        )
                        tool_calls.append(tool_call)

    return tool_calls


def sanitize_run_context(value):
    """
    Recursively replace pydantic-ai RunContext instances with '<RunContext>'.

    This avoids leaking internal context details into recorded function kwargs,
    while keeping the original arguments intact for the actual function call.
    """
    if isinstance(value, RunContext):
        return "<RunContext>"
    if isinstance(value, dict):
        return {k: sanitize_run_context(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        sanitized = [sanitize_run_context(v) for v in value]
        return tuple(sanitized) if isinstance(value, tuple) else sanitized
    if isinstance(value, set):
        return {sanitize_run_context(v) for v in value}

    return value


def patch_llm_model(
    model: Model,
    llm_metric_collection: Optional[str] = None,
    llm_metrics: Optional[List[BaseMetric]] = None,
    llm_prompt: Optional[Prompt] = None,
):
    original_func = model.request
    sig = inspect.signature(original_func)

    try:
        model_name = model.model_name
    except Exception:
        model_name = "unknown"

    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        request = bound.arguments.get("messages", [])

        with Observer(
            span_type="llm",
            func_name="LLM",
            observe_kwargs={"model": model_name},
            metrics=llm_metrics,
            metric_collection=llm_metric_collection,
        ) as observer:
            result = await original_func(*args, **kwargs)
            observer.update_span_properties = (
                lambda llm_span: set_llm_span_attributes(
                    llm_span, request, result, llm_prompt
                )
            )
            observer.result = result
            return result

    model.request = wrapper

    stream_original_func = model.request_stream
    stream_sig = inspect.signature(stream_original_func)

    @asynccontextmanager
    async def stream_wrapper(*args, **kwargs):
        bound = stream_sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        request = bound.arguments.get("messages", [])

        with Observer(
            span_type="llm",
            func_name="LLM",
            observe_kwargs={"model": model_name},
            metrics=llm_metrics,
            metric_collection=llm_metric_collection,
        ) as observer:
            llm_span: LlmSpan = current_span_context.get()
            async with stream_original_func(
                *args, **kwargs
            ) as streamed_response:
                try:
                    yield streamed_response
                    if not llm_span.token_intervals:
                        llm_span.token_intervals = {perf_counter(): "NA"}
                    else:
                        llm_span.token_intervals[perf_counter()] = "NA"
                finally:
                    try:
                        result = streamed_response.get()
                        observer.update_span_properties = (
                            lambda llm_span: set_llm_span_attributes(
                                llm_span, request, result, llm_prompt
                            )
                        )
                        observer.result = result
                    except Exception:
                        pass

    model.request_stream = stream_wrapper


def create_patched_tool(
    func: Callable,
    metrics: Optional[List[BaseMetric]] = None,
    metric_collection: Optional[str] = None,
):
    import asyncio

    original_func = func

    is_async = asyncio.iscoroutinefunction(original_func)

    if is_async:

        @functools.wraps(original_func)
        async def async_wrapper(*args, **kwargs):
            sanitized_args = sanitize_run_context(args)
            sanitized_kwargs = sanitize_run_context(kwargs)
            with Observer(
                span_type="tool",
                func_name=original_func.__name__,
                metrics=metrics,
                metric_collection=metric_collection,
                function_kwargs={"args": sanitized_args, **sanitized_kwargs},
            ) as observer:
                result = await original_func(*args, **kwargs)
                observer.result = result

            return result

        return async_wrapper
    else:

        @functools.wraps(original_func)
        def sync_wrapper(*args, **kwargs):
            sanitized_args = sanitize_run_context(args)
            sanitized_kwargs = sanitize_run_context(kwargs)
            with Observer(
                span_type="tool",
                func_name=original_func.__name__,
                metrics=metrics,
                metric_collection=metric_collection,
                function_kwargs={"args": sanitized_args, **sanitized_kwargs},
            ) as observer:
                result = original_func(*args, **kwargs)
                observer.result = result

            return result

        return sync_wrapper


def update_trace_context(
    trace_name: Optional[str] = None,
    trace_tags: Optional[List[str]] = None,
    trace_metadata: Optional[dict] = None,
    trace_thread_id: Optional[str] = None,
    trace_user_id: Optional[str] = None,
    trace_metric_collection: Optional[str] = None,
    trace_metrics: Optional[List[BaseMetric]] = None,
    trace_input: Optional[Any] = None,
    trace_output: Optional[Any] = None,
):

    current_trace = current_trace_context.get()

    if trace_name:
        current_trace.name = trace_name
    if trace_tags:
        current_trace.tags = trace_tags
    if trace_metadata:
        current_trace.metadata = trace_metadata
    if trace_thread_id:
        current_trace.thread_id = trace_thread_id
    if trace_user_id:
        current_trace.user_id = trace_user_id
    if trace_metric_collection:
        current_trace.metric_collection = trace_metric_collection
    if trace_metrics:
        current_trace.metrics = trace_metrics
    if trace_input:
        current_trace.input = trace_input
    if trace_output:
        current_trace.output = trace_output


def set_llm_span_attributes(
    llm_span: LlmSpan,
    requests: List[ModelRequest],
    result: ModelResponse,
    llm_prompt: Optional[Prompt] = None,
):
    llm_span.prompt = llm_prompt

    input = []
    for request in requests:
        for part in request.parts:
            if isinstance(part, SystemPromptPart):
                input.append({"role": "System", "content": part.content})
            elif isinstance(part, UserPromptPart):
                input.append({"role": "User", "content": part.content})
            elif isinstance(part, ToolCallPart):
                input.append(
                    {
                        "role": "Tool Call",
                        "name": part.tool_name,
                        "content": part.args_as_json_str(),
                    }
                )
            elif isinstance(part, ToolReturnPart):
                input.append(
                    {
                        "role": "Tool Return",
                        "name": part.tool_name,
                        "content": part.model_response_str(),
                    }
                )
    llm_span.input = input

    content = ""
    tool_calls = []
    for part in result.parts:
        if isinstance(part, TextPart):
            content += part.content + "\n"
        elif isinstance(part, ToolCallPart):
            tool_calls.append(
                LlmToolCall(name=part.tool_name, args=part.args_as_dict())
            )
    llm_span.output = LlmOutput(
        role="Assistant", content=content, tool_calls=tool_calls
    )
    llm_span.tools_called = extract_tools_called_from_llm_response(result.parts)


def set_agent_span_attributes(agent_span: AgentSpan, result: AgentRunResult):
    agent_span.tools_called = extract_tools_called(result)
