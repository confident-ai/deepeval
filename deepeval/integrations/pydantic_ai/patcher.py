import functools
import deepeval
from deepeval.tracing.types import LlmOutput, LlmToolCall
from pydantic_ai.agent import AgentRunResult
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.types import AgentSpan, LlmSpan
from deepeval.tracing.tracing import Observer
from typing import List, Callable, Optional, Any
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.metrics.base_metric import BaseMetric
from deepeval.confident.api import get_confident_api_key
from deepeval.integrations.pydantic_ai.otel import instrument_pydantic_ai
from deepeval.telemetry import capture_tracing_integration
from deepeval.prompt import Prompt

try:
    from pydantic_ai.agent import Agent
    from pydantic_ai.models import Model
    from pydantic_ai.messages import (
        ModelResponse,
        ModelRequest,
        ModelResponsePart,
        TextPart,
        ToolCallPart,
        SystemPromptPart,
        ToolReturnPart,
        UserPromptPart,
    )

    pydantic_ai_installed = True
except:
    pydantic_ai_installed = True


def _patch_agent_tool_decorator():
    original_tool = Agent.tool

    @functools.wraps(original_tool)
    def wrapper(
        *args,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        **kwargs
    ):
        # Case 1: Direct decoration - @agent.tool
        if args and callable(args[0]):
            patched_func = _create_patched_tool(
                args[0], metrics, metric_collection
            )
            new_args = (patched_func,) + args[1:]
            return original_tool(*new_args, **kwargs)

        # Case 2: Decoration with arguments - @agent.tool(metrics=..., metric_collection=...)
        else:
            # Return a decorator function that will receive the actual function
            def decorator(func):
                patched_func = _create_patched_tool(
                    func, metrics, metric_collection
                )
                return original_tool(*args, **kwargs)(patched_func)

            return decorator

    Agent.tool = wrapper


def _create_patched_tool(
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
            with Observer(
                span_type="tool",
                func_name=original_func.__name__,
                metrics=metrics,
                metric_collection=metric_collection,
                function_kwargs={"args": args, **kwargs},
            ) as observer:
                result = await original_func(*args, **kwargs)
                observer.result = result

            return result

        return async_wrapper
    else:

        @functools.wraps(original_func)
        def sync_wrapper(*args, **kwargs):
            with Observer(
                span_type="tool",
                func_name=original_func.__name__,
                metrics=metrics,
                metric_collection=metric_collection,
                function_kwargs={"args": args, **kwargs},
            ) as observer:
                result = original_func(*args, **kwargs)
                observer.result = result

            return result

        return sync_wrapper


def _patch_agent_init():
    original_init = Agent.__init__

    @functools.wraps(original_init)
    def wrapper(
        self,
        *args,
        llm_metric_collection: Optional[str] = None,
        llm_metrics: Optional[List[BaseMetric]] = None,
        llm_prompt: Optional[Prompt] = None,
        agent_metric_collection: Optional[str] = None,
        agent_metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ):
        result = original_init(self, *args, **kwargs)
        _patch_llm_model(
            self._model, llm_metric_collection, llm_metrics, llm_prompt
        )  # runtime patch of the model
        _patch_agent_run(agent_metric_collection, agent_metrics)
        return result

    Agent.__init__ = wrapper


def _patch_agent_run(
    agent_metric_collection: Optional[str] = None,
    agent_metrics: Optional[List[BaseMetric]] = None,
):
    original_run = Agent.run

    @functools.wraps(original_run)
    async def wrapper(
        *args,
        trace_metric_collection: Optional[str] = None,
        trace_metrics: Optional[List[BaseMetric]] = None,
        trace_name: Optional[str] = None,
        trace_tags: Optional[List[str]] = None,
        trace_metadata: Optional[dict] = None,
        trace_thread_id: Optional[str] = None,
        trace_user_id: Optional[str] = None,
        **kwargs
    ):
        with Observer(
            span_type="agent",
            func_name="Agent",
            function_kwargs={"input": args[1]},
            metrics=agent_metrics,
            metric_collection=agent_metric_collection,
        ) as observer:
            result = await original_run(*args, **kwargs)
            observer.update_span_properties = (
                lambda agent_span: set_agent_span_attributes(agent_span, result)
            )
            observer.result = result.output

            _update_trace_context(
                trace_name=trace_name,
                trace_tags=trace_tags,
                trace_metadata=trace_metadata,
                trace_thread_id=trace_thread_id,
                trace_user_id=trace_user_id,
                trace_metric_collection=trace_metric_collection,
                trace_metrics=trace_metrics,
                trace_input=args[1],
                trace_output=result.output,
            )

        return result

    Agent.run = wrapper


def _update_trace_context(
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
    current_trace.name = trace_name
    current_trace.tags = trace_tags
    current_trace.metadata = trace_metadata
    current_trace.thread_id = trace_thread_id
    current_trace.user_id = trace_user_id
    current_trace.metric_collection = trace_metric_collection
    current_trace.metrics = trace_metrics
    current_trace.input = trace_input
    current_trace.output = trace_output


def _patch_llm_model(
    model: Model,
    llm_metric_collection: Optional[str] = None,
    llm_metrics: Optional[List[BaseMetric]] = None,
    llm_prompt: Optional[Prompt] = None,
):
    original_func = model.request
    try:
        model_name = model.model_name
    except Exception:
        model_name = "unknown"

    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        with Observer(
            span_type="llm",
            func_name="LLM",
            observe_kwargs={"model": model_name},
            metrics=llm_metrics,
            metric_collection=llm_metric_collection,
        ) as observer:
            result = await original_func(*args, **kwargs)
            request = kwargs.get("messages", [])
            if not request:
                request = args[0]
            observer.update_span_properties = (
                lambda llm_span: set_llm_span_attributes(
                    llm_span, args[0], result, llm_prompt
                )
            )
            observer.result = result
        return result

    model.request = wrapper


def instrument(otel: Optional[bool] = False, api_key: Optional[str] = None):

    if api_key:
        deepeval.login(api_key)

    api_key = get_confident_api_key()

    if not api_key:
        raise ValueError("No api key provided.")

    if otel:
        instrument_pydantic_ai(api_key)
    else:
        with capture_tracing_integration("pydantic_ai"):
            _patch_agent_init()
            _patch_agent_tool_decorator()


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
    llm_span.tools_called = _extract_tools_called_from_llm_response(
        result.parts
    )


def set_agent_span_attributes(agent_span: AgentSpan, result: AgentRunResult):
    agent_span.tools_called = _extract_tools_called(result)


# llm tools called
def _extract_tools_called_from_llm_response(
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
def _extract_tools_called(result: AgentRunResult) -> List[ToolCall]:
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
