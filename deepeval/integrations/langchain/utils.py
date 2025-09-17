from typing import Any, List, Dict, Optional
from langchain_core.outputs import ChatGeneration


def parse_prompts_to_messages(
    prompts: list[str], **kwargs
) -> List[Dict[str, str]]:
    VALID_ROLES = [
        "system",
        "assistant",
        "ai",
        "user",
        "human",
        "tool",
        "function",
    ]

    messages: List[Dict[str, str]] = []
    current_role = None
    current_content: List[str] = []

    for prompt in prompts:
        for line in prompt.splitlines():
            line = line.strip()
            if not line:
                continue

            first_word, sep, rest = line.partition(":")
            role = (
                first_word.lower()
                if sep and first_word.lower() in VALID_ROLES
                else None
            )

            if role:
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role,
                            "content": "\n".join(current_content).strip(),
                        }
                    )
                current_role = role
                current_content = [rest.strip()]
            else:
                if not current_role:
                    current_role = "Human"
                current_content.append(line)

        if current_role and current_content:
            messages.append(
                {
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                }
            )
            current_role, current_content = None, []

    tools = kwargs.get("invocation_params", {}).get("tools", None)
    if tools and isinstance(tools, list):
        for tool in tools:
            messages.append({"role": "Tool Input", "content": str(tool)})

    return messages


def convert_chat_generation_to_string(gen: ChatGeneration) -> str:
    return gen.message.pretty_repr()


def prepare_dict(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def safe_extract_token_usage(
    response_metadata: dict[str, Any],
) -> tuple[int, int]:
    prompt_tokens, completion_tokens = 0, 0
    token_usage = response_metadata.get("token_usage")
    if token_usage and isinstance(token_usage, dict):
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)

    return prompt_tokens, completion_tokens


def extract_name(serialized: dict[str, Any], **kwargs: Any) -> str:
    if "name" in kwargs and kwargs["name"]:
        return kwargs["name"]

    if "name" in serialized:
        return serialized["name"]

    return "Agent"


def safe_extract_model_name(
    metadata: dict[str, Any], **kwargs: Any
) -> Optional[str]:
    if kwargs and isinstance(kwargs, dict):
        invocation_params = kwargs.get("invocation_params")
        if invocation_params:
            model = invocation_params.get("model")
            if model:
                return model

    if metadata:
        ls_model_name = metadata.get("ls_model_name")
        if ls_model_name:
            return ls_model_name

    return None


from typing import Any, List, Dict, Optional, Union, Literal, Callable
from langchain_core.outputs import ChatGeneration
from time import perf_counter
import uuid
from rich.progress import Progress
from deepeval.tracing.tracing import Observer

from deepeval.metrics import BaseMetric
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    SpanType,
    ToolSpan,
    TraceSpanStatus,
)


def enter_current_context(
    span_type: Optional[
        Union[Literal["agent", "llm", "retriever", "tool"], str]
    ],
    func_name: str,
    metrics: Optional[Union[List[str], List[BaseMetric]]] = None,
    metric_collection: Optional[str] = None,
    observe_kwargs: Optional[Dict[str, Any]] = None,
    function_kwargs: Optional[Dict[str, Any]] = None,
    progress: Optional[Progress] = None,
    pbar_callback_id: Optional[int] = None,
    uuid_str: Optional[str] = None,
) -> BaseSpan:
    start_time = perf_counter()
    observe_kwargs = observe_kwargs or {}
    function_kwargs = function_kwargs or {}

    name = observe_kwargs.get("name", func_name)
    prompt = observe_kwargs.get("prompt", None)
    uuid_str = uuid_str or str(uuid.uuid4())

    parent_span = current_span_context.get()
    trace_uuid: Optional[str] = None
    parent_uuid: Optional[str] = None

    if parent_span:
        parent_uuid = parent_span.uuid
        trace_uuid = parent_span.trace_uuid
    else:
        current_trace = current_trace_context.get()
        if current_trace:
            trace_uuid = current_trace.uuid
        else:
            trace = trace_manager.start_new_trace(
                metric_collection=metric_collection
            )
            trace_uuid = trace.uuid
            current_trace_context.set(trace)

    span_kwargs = {
        "uuid": uuid_str,
        "trace_uuid": trace_uuid,
        "parent_uuid": parent_uuid,
        "start_time": start_time,
        "end_time": None,
        "status": TraceSpanStatus.SUCCESS,
        "children": [],
        "name": name,
        "input": None,
        "output": None,
        "metrics": metrics,
        "metric_collection": metric_collection,
    }

    if span_type == SpanType.AGENT.value:
        available_tools = observe_kwargs.get("available_tools", [])
        agent_handoffs = observe_kwargs.get("agent_handoffs", [])
        span_instance = AgentSpan(
            **span_kwargs,
            available_tools=available_tools,
            agent_handoffs=agent_handoffs,
        )
    elif span_type == SpanType.LLM.value:
        model = observe_kwargs.get("model", None)
        c_in = observe_kwargs.get("cost_per_input_token", None)
        c_out = observe_kwargs.get("cost_per_output_token", None)
        span_instance = LlmSpan(
            **span_kwargs,
            model=model,
            cost_per_input_token=c_in,
            cost_per_output_token=c_out,
        )
    elif span_type == SpanType.RETRIEVER.value:
        embedder = observe_kwargs.get("embedder", None)
        span_instance = RetrieverSpan(**span_kwargs, embedder=embedder)
    elif span_type == SpanType.TOOL.value:
        span_instance = ToolSpan(**span_kwargs, **observe_kwargs)
    else:
        span_instance = BaseSpan(**span_kwargs)

    # Set input and prompt at entry
    span_instance.input = trace_manager.mask(function_kwargs)
    if isinstance(span_instance, LlmSpan) and prompt:
        span_instance.prompt = prompt

    trace_manager.add_span(span_instance)
    trace_manager.add_span_to_trace(span_instance)

    if (
        parent_span
        and getattr(parent_span, "progress", None) is not None
        and getattr(parent_span, "pbar_callback_id", None) is not None
    ):
        progress = parent_span.progress
        pbar_callback_id = parent_span.pbar_callback_id

    if progress is not None and pbar_callback_id is not None:
        span_instance.progress = progress
        span_instance.pbar_callback_id = pbar_callback_id

    current_span_context.set(span_instance)

    # return {
    #     "uuid": uuid_str,
    #     "progress": progress,
    #     "pbar_callback_id": pbar_callback_id,
    # }

    return span_instance


def exit_current_context(
    uuid_str: str,
    result: Any = None,
    update_span_properties: Optional[Callable[[BaseSpan], None]] = None,
    progress: Optional[Progress] = None,
    pbar_callback_id: Optional[int] = None,
    exc_type: Optional[type] = None,
    exc_val: Optional[BaseException] = None,
    exc_tb: Optional[Any] = None,
) -> None:
    end_time = perf_counter()

    current_span = current_span_context.get()

    if not current_span or current_span.uuid != uuid_str:
        print(
            f"Error: Current span in context does not match the span being exited. Expected UUID: {uuid_str}, Got: {current_span.uuid if current_span else 'None'}"
        )
        return

    current_span.end_time = end_time
    if exc_type is not None:
        current_span.status = TraceSpanStatus.ERRORED
        current_span.error = str(exc_val)
    else:
        current_span.status = TraceSpanStatus.SUCCESS

    if update_span_properties is not None:
        update_span_properties(current_span)

    # Only set output on exit
    if current_span.output is None:
        current_span.output = trace_manager.mask(result)

    # Prefer provided progress info, but fallback to span fields if missing
    if progress is None and getattr(current_span, "progress", None) is not None:
        progress = current_span.progress
    if (
        pbar_callback_id is None
        and getattr(current_span, "pbar_callback_id", None) is not None
    ):
        pbar_callback_id = current_span.pbar_callback_id

    trace_manager.remove_span(uuid_str)
    if current_span.parent_uuid:
        parent_span = trace_manager.get_span_by_uuid(current_span.parent_uuid)
        if parent_span:
            current_span_context.set(parent_span)
        else:
            current_span_context.set(None)
    else:
        current_trace = current_trace_context.get()
        if current_span.status == TraceSpanStatus.ERRORED and current_trace:
            current_trace.status = TraceSpanStatus.ERRORED
        if current_trace and current_trace.uuid == current_span.trace_uuid:
            other_active_spans = [
                span
                for span in trace_manager.active_spans.values()
                if span.trace_uuid == current_span.trace_uuid
            ]
            if not other_active_spans:
                trace_manager.end_trace(current_span.trace_uuid)
                current_trace_context.set(None)

        current_span_context.set(None)

    if progress is not None and pbar_callback_id is not None:
        progress.update(pbar_callback_id, advance=1)
