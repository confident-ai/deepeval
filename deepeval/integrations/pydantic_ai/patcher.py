import functools
from pydantic_ai.agent import AgentRunResult
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.types import AgentSpan, LlmSpan
from deepeval.tracing.tracing import Observer
from typing import List, Callable, Optional
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.metrics.base_metric import BaseMetric
try:
    from pydantic_ai.agent import Agent
    from pydantic_ai.models import Model
    from pydantic_ai.messages import ModelResponse, ModelRequest, ModelResponsePart
    pydantic_ai_installed = True
except:
    pydantic_ai_installed = True

def _patch_agent_tool_decorator():
    original_tool = Agent.tool
    
    @functools.wraps(original_tool)
    def wrapper(*args, metrics: Optional[List[BaseMetric]] = None, metric_collection: Optional[str] = None, **kwargs):
        # Case 1: Direct decoration - @agent.tool
        if args and callable(args[0]):
            patched_func = _create_patched_tool(args[0], metrics, metric_collection)
            new_args = (patched_func,) + args[1:]
            return original_tool(*new_args, **kwargs)
        
        # Case 2: Decoration with arguments - @agent.tool(metrics=..., metric_collection=...)
        else:
            # Return a decorator function that will receive the actual function
            def decorator(func):
                patched_func = _create_patched_tool(func, metrics, metric_collection)
                return original_tool(*args, **kwargs)(patched_func)
            return decorator
    
    Agent.tool = wrapper

def _create_patched_tool(func: Callable, metrics: Optional[List[BaseMetric]] = None, metric_collection: Optional[str] = None):
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
    def wrapper(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        _patch_llm_model(self._model) # runtime patch of the model
        return result

    Agent.__init__ = wrapper

def _patch_agent_run():
    original_run = Agent.run

    @functools.wraps(original_run)
    async def wrapper(
        *args,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
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
            metrics=metrics,
            metric_collection=metric_collection,
        ) as observer:
            result = await original_run(*args, **kwargs)
            observer.update_span_properties = (
                lambda agent_span: set_agent_span_attributes(
                    agent_span, result
                )
            )
            observer.result = result.output
            current_trace = current_trace_context.get()
            
            current_trace.input = args[1]
            current_trace.output = result.output

            current_trace.name = trace_name
            current_trace.tags = trace_tags
            current_trace.metadata = trace_metadata
            current_trace.thread_id = trace_thread_id
            current_trace.user_id = trace_user_id
            
        return result

    Agent.run = wrapper

def _patch_llm_model(model: Model):
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
        ) as observer:
            result = await original_func(*args, **kwargs)
            request = kwargs.get("messages", [])
            if not request:
                request = args[0]
            observer.update_span_properties = (
                lambda llm_span: set_llm_span_attributes(llm_span, args[0], result)
            )
            observer.result = result
        return result
    model.request = wrapper

def patch_all():
    _patch_agent_init()
    _patch_agent_run()
    _patch_agent_tool_decorator()

def set_llm_span_attributes(llm_span: LlmSpan, request: List[ModelRequest], result: ModelResponse):
    llm_span.input = [r.parts for r in request] # debug more on this
    llm_span.output = result.parts
    llm_span.tools_called = _extract_tools_called_from_llm_response(result.parts)

def set_agent_span_attributes(agent_span: AgentSpan, result: AgentRunResult):
    agent_span.tools_called = _extract_tools_called(result)

# llm tools called
def _extract_tools_called_from_llm_response(result: List[ModelResponsePart]) -> List[ToolCall]:
    tool_calls = []
    
    # Loop through each ModelResponsePart
    for part in result:
        # Look for parts with part_kind="tool-call"
        if hasattr(part, 'part_kind') and part.part_kind == "tool-call":
            # Extract tool name and args from the ToolCallPart
            tool_name = part.tool_name
            input_parameters = part.args_as_dict() if hasattr(part, 'args_as_dict') else None
            
            # Create and append ToolCall object
            tool_call = ToolCall(
                name=tool_name,
                input_parameters=input_parameters
            )
            tool_calls.append(tool_call)
    
    return tool_calls

#TODO: llm tools called (reposne is present next message)
def _extract_tools_called(result: AgentRunResult) -> List[ToolCall]:
    tool_calls = []
    
    # Access the message history from the _state
    message_history = result._state.message_history
    
    # Scan through all messages in the history
    for message in message_history:
        # Check if this is a ModelResponse (kind="response")
        if hasattr(message, 'kind') and message.kind == "response":
            # For ModelResponse messages, check each part
            if hasattr(message, 'parts'):
                for part in message.parts:
                    # Look for parts with part_kind="tool-call"
                    if hasattr(part, 'part_kind') and part.part_kind == "tool-call":
                        # Extract tool name and args from the ToolCallPart
                        tool_name = part.tool_name
                        input_parameters = part.args_as_dict() if hasattr(part, 'args_as_dict') else None
                        
                        # Create and append ToolCall object
                        tool_call = ToolCall(
                            name=tool_name,
                            input_parameters=input_parameters
                        )
                        tool_calls.append(tool_call)
    
    return tool_calls
    