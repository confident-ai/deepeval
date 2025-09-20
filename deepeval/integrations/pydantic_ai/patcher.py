# import inspect
# import functools
# import warnings
# from typing import List, Callable, Optional, Any
# from deepeval.tracing.types import LlmOutput, LlmToolCall
# from pydantic_ai.agent import AgentRunResult
# from deepeval.tracing.context import current_trace_context
# from deepeval.tracing.types import AgentSpan, LlmSpan
# from deepeval.tracing.tracing import Observer
# from deepeval.test_case.llm_test_case import ToolCall
# from deepeval.metrics.base_metric import BaseMetric
# from deepeval.confident.api import get_confident_api_key
# from deepeval.integrations.pydantic_ai.otel import instrument_pydantic_ai
# from deepeval.telemetry import capture_tracing_integration
# from deepeval.prompt import Prompt
# import deepeval
# # from contextvars import ContextVar

# try:
#     from pydantic_ai.agent import Agent
#     from pydantic_ai.models import Model
#     from pydantic_ai.messages import (
#         ModelResponse,
#         ModelRequest,
#         ModelResponsePart,
#         TextPart,
#         ToolCallPart,
#         SystemPromptPart,
#         ToolReturnPart,
#         UserPromptPart,
#     )
#     from pydantic_ai._run_context import RunContext
#     from deepeval.integrations.pydantic_ai.utils import (
#         extract_tools_called_from_llm_response,
#         extract_tools_called,
#         sanitize_run_context,
#     )

#     pydantic_ai_installed = True
# except:
#     pydantic_ai_installed = True

# # _IN_RUN_SYNC = ContextVar("deepeval_in_run_sync", default=False)
# # _INSTRUMENTED = False


import warnings
from typing import Optional


def instrument(otel: Optional[bool] = False, api_key: Optional[str] = None):
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Please deepeval.integrations.pydantic_ai.Agent to instrument instead.
    """
    warnings.warn(
        "The 'instrument_pydantic_ai()' function is deprecated and will be removed in a future version. "
        "Please use deepeval.integrations.pydantic_ai.Agent to instrument instead. Refer to the documentation [link]",  # TODO: add the link,
        UserWarning,
        stacklevel=2,
    )

    # Don't execute the original functionality
    return

    # Original code below (commented out to prevent execution)
    # global _INSTRUMENTED
    # if api_key:
    #     deepeval.login(api_key)
    #
    # api_key = get_confident_api_key()
    #
    # if not api_key:
    #     raise ValueError("No api key provided.")
    #
    # if otel:
    #     instrument_pydantic_ai(api_key)
    # else:
    #     with capture_tracing_integration("pydantic_ai"):
    #         if _INSTRUMENTED:
    #             return
    #         _patch_agent_init()
    #         _patch_agent_tool_decorator()
    #         _INSTRUMENTED = True


# ################### Init Patches ###################


# # def _patch_agent_init():
# #     original_init = Agent.__init__

# #     @functools.wraps(original_init)
# #     def wrapper(
# #         *args,
# #         llm_metric_collection: Optional[str] = None,
# #         llm_metrics: Optional[List[BaseMetric]] = None,
# #         llm_prompt: Optional[Prompt] = None,
# #         agent_metric_collection: Optional[str] = None,
# #         agent_metrics: Optional[List[BaseMetric]] = None,
# #         name: Optional[str] = None,
# #         tags: Optional[List[str]] = None,
# #         metadata: Optional[dict] = None,
# #         thread_id: Optional[str] = None,
# #         user_id: Optional[str] = None,
# #         metric_collection: Optional[str] = None,
# #         metrics: Optional[List[BaseMetric]] = None,
# #         **kwargs
# #     ):
# #         result = original_init(*args, **kwargs)
# #         _patch_llm_model(args[0]._model, llm_metric_collection, llm_metrics, llm_prompt)  # runtime patch of the model
# #         _patch_agent_run(
# #             agent=args[0],
# #             agent_metric_collection=agent_metric_collection,
# #             agent_metrics=agent_metrics,
# #             init_trace_name=name,
# #             init_trace_tags=tags,
# #             init_trace_metadata=metadata,
# #             init_trace_thread_id=thread_id,
# #             init_trace_user_id=user_id,
# #             init_trace_metric_collection=metric_collection,
# #             init_trace_metrics=metrics,
# #         )
# #         _patch_agent_run_sync(
# #             agent=args[0],
# #             agent_metric_collection=agent_metric_collection,
# #             agent_metrics=agent_metrics,
# #             init_trace_name=name,
# #             init_trace_tags=tags,
# #             init_trace_metadata=metadata,
# #             init_trace_thread_id=thread_id,
# #             init_trace_user_id=user_id,
# #             init_trace_metric_collection=metric_collection,
# #             init_trace_metrics=metrics,
# #         )
# #         return result

# #     Agent.__init__ = wrapper


# # def _patch_agent_tool_decorator():
# #     original_tool = Agent.tool

# #     @functools.wraps(original_tool)
# #     def wrapper(
# #         *args,
# #         metrics: Optional[List[BaseMetric]] = None,
# #         metric_collection: Optional[str] = None,
# #         **kwargs
# #     ):
# #         # Case 1: Direct decoration - @agent.tool
# #         if args and callable(args[0]):
# #             patched_func = _create_patched_tool(
# #                 args[0], metrics, metric_collection
# #             )
# #             new_args = (patched_func,) + args[1:]
# #             return original_tool(*new_args, **kwargs)

# #         # Case 2: Decoration with arguments - @agent.tool(metrics=..., metric_collection=...)
# #         else:
# #             # Return a decorator function that will receive the actual function
# #             def decorator(func):
# #                 patched_func = _create_patched_tool(
# #                     func, metrics, metric_collection
# #                 )
# #                 return original_tool(*args, **kwargs)(patched_func)

# #             return decorator

# #     Agent.tool = wrapper


# ################### Runtime Patches ###################


# # def _patch_agent_run_sync(
# #     agent: Agent,
# #     agent_metric_collection: Optional[str] = None,
# #     agent_metrics: Optional[List[BaseMetric]] = None,
# #     init_trace_name: Optional[str] = None,
# #     init_trace_tags: Optional[List[str]] = None,
# #     init_trace_metadata: Optional[dict] = None,
# #     init_trace_thread_id: Optional[str] = None,
# #     init_trace_user_id: Optional[str] = None,
# #     init_trace_metric_collection: Optional[str] = None,
# #     init_trace_metrics: Optional[List[BaseMetric]] = None,
# # ):
# #     original_run_sync = agent.run_sync

# #     @functools.wraps(original_run_sync)
# #     def wrapper(
# #         *args,
# #         metric_collection: Optional[str] = None,
# #         metrics: Optional[List[BaseMetric]] = None,
# #         name: Optional[str] = None,
# #         tags: Optional[List[str]] = None,
# #         metadata: Optional[dict] = None,
# #         thread_id: Optional[str] = None,
# #         user_id: Optional[str] = None,
# #         **kwargs
# #     ):

# #         sig = inspect.signature(original_run_sync)
# #         bound = sig.bind_partial(*args, **kwargs)
# #         bound.apply_defaults()
# #         input = bound.arguments.get("user_prompt", None)

# #         with Observer(
# #             span_type="agent",
# #             func_name="Agent",
# #             function_kwargs={"input": input},
# #             metrics=agent_metrics,
# #             metric_collection=agent_metric_collection,
# #         ) as observer:

# #             token = _IN_RUN_SYNC.set(True)
# #             try:
# #                 result = original_run_sync(*args, **kwargs)
# #             finally:
# #                 _IN_RUN_SYNC.reset(token)

# #             observer.update_span_properties = (
# #                 lambda agent_span: set_agent_span_attributes(agent_span, result)
# #             )
# #             observer.result = result.output

# #             _update_trace_context(
# #                 trace_name=init_trace_name if init_trace_name else name,
# #                 trace_tags=init_trace_tags if init_trace_tags else tags,
# #                 trace_metadata=init_trace_metadata if init_trace_metadata else metadata,
# #                 trace_thread_id=init_trace_thread_id if init_trace_thread_id else thread_id,
# #                 trace_user_id=init_trace_user_id if init_trace_user_id else user_id,
# #                 trace_metric_collection=init_trace_metric_collection if init_trace_metric_collection else metric_collection,
# #                 trace_metrics=init_trace_metrics if init_trace_metrics else metrics,
# #                 trace_input=input,
# #                 trace_output=result.output,
# #             )

# #         return result

# #     agent.run_sync = wrapper


# # def _patch_agent_run(
# #     agent: Agent,
# #     agent_metric_collection: Optional[str] = None,
# #     agent_metrics: Optional[List[BaseMetric]] = None,
# #     init_trace_name: Optional[str] = None,
# #     init_trace_tags: Optional[List[str]] = None,
# #     init_trace_metadata: Optional[dict] = None,
# #     init_trace_thread_id: Optional[str] = None,
# #     init_trace_user_id: Optional[str] = None,
# #     init_trace_metric_collection: Optional[str] = None,
# #     init_trace_metrics: Optional[List[BaseMetric]] = None,
# # ):
# #     original_run = agent.run

# #     @functools.wraps(original_run)
# #     async def wrapper(
# #         *args,
# #         metric_collection: Optional[str] = None,
# #         metrics: Optional[List[BaseMetric]] = None,
# #         name: Optional[str] = None,
# #         tags: Optional[List[str]] = None,
# #         metadata: Optional[dict] = None,
# #         thread_id: Optional[str] = None,
# #         user_id: Optional[str] = None,
# #         **kwargs
# #     ):
# #         sig = inspect.signature(original_run)
# #         bound = sig.bind_partial(*args, **kwargs)
# #         bound.apply_defaults()
# #         input = bound.arguments.get("user_prompt", None)

# #         in_sync = _IN_RUN_SYNC.get()
# #         with Observer(
# #             span_type="agent" if not in_sync else "custom",
# #             func_name="Agent" if not in_sync else "run",
# #             function_kwargs={"input": input},
# #             metrics=agent_metrics if not in_sync else None,
# #             metric_collection=agent_metric_collection if not in_sync else None,
# #         ) as observer:
# #             print(args)
# #             print(kwargs)
# #             result = await original_run(*args, **kwargs)
# #             observer.update_span_properties = (
# #                 lambda agent_span: set_agent_span_attributes(agent_span, result)
# #             )
# #             observer.result = result.output

# #             _update_trace_context(
# #                 trace_name=init_trace_name if init_trace_name else name,
# #                 trace_tags=init_trace_tags if init_trace_tags else tags,
# #                 trace_metadata=init_trace_metadata if init_trace_metadata else metadata,
# #                 trace_thread_id=init_trace_thread_id if init_trace_thread_id else thread_id,
# #                 trace_user_id=init_trace_user_id if init_trace_user_id else user_id,
# #                 trace_metric_collection=init_trace_metric_collection if init_trace_metric_collection else metric_collection,
# #                 trace_metrics=init_trace_metrics if init_trace_metrics else metrics,
# #                 trace_input=input,
# #                 trace_output=result.output,
# #             )

# #         return result

# #     agent.run = wrapper


# def patch_llm_model(
#     model: Model,
#     llm_metric_collection: Optional[str] = None,
#     llm_metrics: Optional[List[BaseMetric]] = None,
#     llm_prompt: Optional[Prompt] = None,
# ):
#     original_func = model.request
#     sig = inspect.signature(original_func)

#     try:
#         model_name = model.model_name
#     except Exception:
#         model_name = "unknown"

#     @functools.wraps(original_func)
#     async def wrapper(*args, **kwargs):
#         bound = sig.bind_partial(*args, **kwargs)
#         bound.apply_defaults()
#         request = bound.arguments.get("messages", [])

#         with Observer(
#             span_type="llm",
#             func_name="LLM",
#             observe_kwargs={"model": model_name},
#             metrics=llm_metrics,
#             metric_collection=llm_metric_collection,
#         ) as observer:
#             result = await original_func(*args, **kwargs)
#             observer.update_span_properties = (
#                 lambda llm_span: set_llm_span_attributes(
#                     llm_span, request, result, llm_prompt
#                 )
#             )
#             observer.result = result
#             return result

#     model.request = wrapper


# ################### Helper Functions ###################


# def create_patched_tool(
#     func: Callable,
#     metrics: Optional[List[BaseMetric]] = None,
#     metric_collection: Optional[str] = None,
# ):
#     import asyncio

#     original_func = func

#     is_async = asyncio.iscoroutinefunction(original_func)

#     if is_async:

#         @functools.wraps(original_func)
#         async def async_wrapper(*args, **kwargs):
#             sanitized_args = sanitize_run_context(args)
#             sanitized_kwargs = sanitize_run_context(kwargs)
#             with Observer(
#                 span_type="tool",
#                 func_name=original_func.__name__,
#                 metrics=metrics,
#                 metric_collection=metric_collection,
#                 function_kwargs={"args": sanitized_args, **sanitized_kwargs},
#             ) as observer:
#                 result = await original_func(*args, **kwargs)
#                 observer.result = result

#             return result

#         return async_wrapper
#     else:

#         @functools.wraps(original_func)
#         def sync_wrapper(*args, **kwargs):
#             sanitized_args = sanitize_run_context(args)
#             sanitized_kwargs = sanitize_run_context(kwargs)
#             with Observer(
#                 span_type="tool",
#                 func_name=original_func.__name__,
#                 metrics=metrics,
#                 metric_collection=metric_collection,
#                 function_kwargs={"args": sanitized_args, **sanitized_kwargs},
#             ) as observer:
#                 result = original_func(*args, **kwargs)
#                 observer.result = result

#             return result

#         return sync_wrapper


# def update_trace_context(
#     trace_name: Optional[str] = None,
#     trace_tags: Optional[List[str]] = None,
#     trace_metadata: Optional[dict] = None,
#     trace_thread_id: Optional[str] = None,
#     trace_user_id: Optional[str] = None,
#     trace_metric_collection: Optional[str] = None,
#     trace_metrics: Optional[List[BaseMetric]] = None,
#     trace_input: Optional[Any] = None,
#     trace_output: Optional[Any] = None,
# ):

#     current_trace = current_trace_context.get()

#     if trace_name:
#         current_trace.name = trace_name
#     if trace_tags:
#         current_trace.tags = trace_tags
#     if trace_metadata:
#         current_trace.metadata = trace_metadata
#     if trace_thread_id:
#         current_trace.thread_id = trace_thread_id
#     if trace_user_id:
#         current_trace.user_id = trace_user_id
#     if trace_metric_collection:
#         current_trace.metric_collection = trace_metric_collection
#     if trace_metrics:
#         current_trace.metrics = trace_metrics
#     if trace_input:
#         current_trace.input = trace_input
#     if trace_output:
#         current_trace.output = trace_output


# def set_llm_span_attributes(
#     llm_span: LlmSpan,
#     requests: List[ModelRequest],
#     result: ModelResponse,
#     llm_prompt: Optional[Prompt] = None,
# ):
#     llm_span.prompt = llm_prompt

#     input = []
#     for request in requests:
#         for part in request.parts:
#             if isinstance(part, SystemPromptPart):
#                 input.append({"role": "System", "content": part.content})
#             elif isinstance(part, UserPromptPart):
#                 input.append({"role": "User", "content": part.content})
#             elif isinstance(part, ToolCallPart):
#                 input.append(
#                     {
#                         "role": "Tool Call",
#                         "name": part.tool_name,
#                         "content": part.args_as_json_str(),
#                     }
#                 )
#             elif isinstance(part, ToolReturnPart):
#                 input.append(
#                     {
#                         "role": "Tool Return",
#                         "name": part.tool_name,
#                         "content": part.model_response_str(),
#                     }
#                 )
#     llm_span.input = input

#     content = ""
#     tool_calls = []
#     for part in result.parts:
#         if isinstance(part, TextPart):
#             content += part.content + "\n"
#         elif isinstance(part, ToolCallPart):
#             tool_calls.append(
#                 LlmToolCall(name=part.tool_name, args=part.args_as_dict())
#             )
#     llm_span.output = LlmOutput(
#         role="Assistant", content=content, tool_calls=tool_calls
#     )
#     llm_span.tools_called = extract_tools_called_from_llm_response(result.parts)


# def set_agent_span_attributes(agent_span: AgentSpan, result: AgentRunResult):
#     agent_span.tools_called = extract_tools_called(result)
