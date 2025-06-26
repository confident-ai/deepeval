from time import perf_counter
import inspect
from typing import Any, Dict, Optional, TypeVar

try:
    from llama_index_instrumentation.base import BaseEvent
    from llama_index_instrumentation.dispatcher import Dispatcher
    from llama_index.core.instrumentation.event_handlers.base import (
        BaseEventHandler,
    )
    from llama_index.core.instrumentation.span_handlers.base import (
        BaseSpanHandler,
    )
    from llama_index.core.agent.workflow.workflow_events import (
        ToolCall,
        ToolCallResult,
    )
    from llama_index.core.instrumentation.events.llm import (
        LLMChatStartEvent,
        LLMChatEndEvent,
    )
    from llama_index.core.instrumentation.events.span import SpanDropEvent
    from llama_index.core.instrumentation.span.base import BaseSpan

    T = TypeVar("T", bound=BaseSpan)
    llama_index_installed = True
except:
    llama_index_installed = False


def is_llama_index_installed():
    if not llama_index_installed:
        raise ImportError(
            "llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice."
        )


from deepeval.tracing.types import (
    LlmSpan,
    LlmAttributes,
    RetrieverSpan,
    ToolSpan,
    TraceSpanStatus,
)
from deepeval.tracing import trace_manager

# globals
active_trace_uuid: Optional[str] = None
# span mapping
llm_span_dict: Dict[str, LlmSpan] = {}
tool_span_dict: Dict[str, ToolSpan] = {}
retriever_span_dict: Dict[str, RetrieverSpan] = {}


# might be used for debugging
def serialize(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)  # fallback


class LLamaIndexEventHandler(BaseEventHandler):
    """LlamaIndex custom EventHandler."""

    def __init__(self):
        is_llama_index_installed()
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLamaIndexEventHandler"

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        """Logic for handling event."""

        global active_trace_uuid
        global llm_span_dict
        global tool_span_dict
        global retriever_span_dict

        if isinstance(event, LLMChatStartEvent):
            if not active_trace_uuid:
                active_trace_uuid = trace_manager.start_new_trace().uuid

            input_messages = []
            for msg in event.messages:
                role = msg.role.value
                content = " ".join(
                    block.text
                    for block in msg.blocks
                    if getattr(block, "block_type", None) == "text"
                ).strip()
                input_messages.append({"role": role, "content": content})

            llm_span_dict[event.span_id] = LlmSpan(
                name="confident_llama_index_llm_span",
                uuid=event.id_,
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=active_trace_uuid,
                parent_uuid=None,
                start_time=perf_counter(),
                model=getattr(event, "model_dict", {}).get("model", "unknown"),
                attributes=LlmAttributes(input=input_messages, output=""),
            )

        if isinstance(event, LLMChatEndEvent):

            if event.span_id in llm_span_dict:
                llm_span = llm_span_dict[event.span_id]
                try:
                    response = event.response.message.blocks[0].text
                except:
                    response = ""

                # Update the span by reference
                llm_span.end_time = perf_counter()
                llm_span.status = TraceSpanStatus.SUCCESS
                llm_span.attributes.output = response

                # Update the dictionary with the modified span
                llm_span_dict[event.span_id] = llm_span

        # this is the last event, so we can add all the spans to the trace
        if isinstance(event, SpanDropEvent):
            # add all spans in all the dictionaries to the trace
            for span in llm_span_dict.values():
                trace_manager.add_span_to_trace(span)
            for span in tool_span_dict.values():
                trace_manager.add_span_to_trace(span)
            for span in retriever_span_dict.values():
                trace_manager.add_span_to_trace(span)

            trace_manager.end_trace(active_trace_uuid)

            # reset the dictionaries
            llm_span_dict = {}
            tool_span_dict = {}
            retriever_span_dict = {}
            active_trace_uuid = None


# used for tool calls
class LLamaIndexSpanHandler(BaseSpanHandler):

    def __init__(self):
        is_llama_index_installed()
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLamaIndexSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[T]:

        global active_trace_uuid
        global tool_span_dict

        if not active_trace_uuid:
            active_trace_uuid = trace_manager.start_new_trace().uuid

        _ev = bound_args.arguments.get("ev")
        if _ev is not None and isinstance(_ev, ToolCall):

            tool_span_dict[_ev.tool_id] = ToolSpan(
                uuid=_ev.tool_id,
                name=_ev.tool_name,
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                parent_uuid=None,
                trace_uuid=active_trace_uuid,
                start_time=perf_counter(),
                input=_ev.tool_kwargs,
            )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[T]:

        global tool_span_dict

        _ev = bound_args.arguments.get("ev")
        if _ev is not None and isinstance(_ev, ToolCallResult):
            if _ev.tool_id in tool_span_dict:
                tool_span = tool_span_dict[_ev.tool_id]
                tool_span.end_time = perf_counter()
                tool_span.status = TraceSpanStatus.SUCCESS
                tool_span.output = _ev.tool_output.content
                tool_span_dict[_ev.tool_id] = tool_span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        pass


def instrument_llama_index(dispatcher: Dispatcher):
    is_llama_index_installed()
    dispatcher.add_event_handler(LLamaIndexEventHandler())
    dispatcher.add_span_handler(LLamaIndexSpanHandler())
