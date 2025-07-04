from typing import Any, Dict, Optional
import inspect
from time import perf_counter
import uuid
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.attributes import LlmAttributes, ToolAttributes
from deepeval.tracing.types import BaseSpan, LlmSpan, ToolSpan, TraceSpanStatus

try:
    from llama_index.core.instrumentation.events.base import BaseEvent
    from llama_index.core.instrumentation.event_handlers.base import (
        BaseEventHandler,
    )
    from llama_index.core.instrumentation.span_handlers.base import (
        BaseSpanHandler,
    )
    from llama_index.core.instrumentation.span.base import (
        BaseSpan as LlamaIndexBaseSpan,
    )
    from llama_index.core.instrumentation.events.llm import (
        LLMChatStartEvent,
        LLMChatEndEvent,
    )
    from llama_index.core.tools.function_tool import AsyncBaseTool
    from llama_index_instrumentation.dispatcher import Dispatcher

    llama_index_installed = True
except:
    llama_index_installed = False


def is_llama_index_installed():
    if not llama_index_installed:
        raise ImportError(
            "llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice."
        )


class LLamaIndexHandler(BaseEventHandler, BaseSpanHandler):
    active_trace_uuid: Optional[str] = None
    open_ai_astream_to_llm_span_map: Dict[str, str] = {}

    def __init__(self):
        capture_tracing_integration("llama-index")
        is_llama_index_installed()
        super().__init__()

    def handle(self, event: BaseEvent, **kwargs) -> Any:

        if isinstance(event, LLMChatStartEvent):
            # prepare the input messages
            input_messages = []
            for msg in event.messages:
                role = msg.role.value
                content = " ".join(
                    block.text
                    for block in msg.blocks
                    if getattr(block, "block_type", None) == "text"
                ).strip()
                input_messages.append({"role": role, "content": content})

            # create the span
            llm_span = LlmSpan(
                name="ConfidentLLMSpan",
                uuid=str(uuid.uuid4()),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=self.active_trace_uuid,
                parent_uuid=event.span_id,
                start_time=perf_counter(),
                model=getattr(event, "model_dict", {}).get(
                    "model", "unknown"
                ),  # check the model name not coming in this option
                attributes=LlmAttributes(input=input_messages, output=""),
            )
            trace_manager.add_span(llm_span)
            trace_manager.add_span_to_trace(llm_span)

            # maintaining this since span exits before end llm chat end event
            self.open_ai_astream_to_llm_span_map[event.span_id] = llm_span.uuid

        if isinstance(event, LLMChatEndEvent):
            llm_span_uuid = self.open_ai_astream_to_llm_span_map.get(
                event.span_id
            )
            if llm_span_uuid:
                llm_span = trace_manager.get_span_by_uuid(llm_span_uuid)
                if llm_span:
                    llm_span.status = TraceSpanStatus.SUCCESS
                    llm_span.end_time = perf_counter()
                    llm_span.set_attributes(
                        LlmAttributes(
                            input=llm_span.attributes.input,
                            output=event.response.message.blocks[0].text,
                        )
                    )  # only takes the message response ouput, but what if the response is a tool?
                    trace_manager.remove_span(llm_span.uuid)
                    del self.open_ai_astream_to_llm_span_map[event.span_id]

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[LlamaIndexBaseSpan]:
        if parent_span_id is None:
            self.active_trace_uuid = trace_manager.start_new_trace().uuid

        base_span = BaseSpan(
            uuid=id_,
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_uuid,
            parent_uuid=parent_span_id,
            start_time=perf_counter(),
            name=instance.__class__.__name__ if instance else None,
            input=bound_args.arguments,
        )

        trace_manager.add_span(base_span)
        trace_manager.add_span_to_trace(base_span)

        if isinstance(instance, AsyncBaseTool):
            tool_span = ToolSpan(
                uuid=str(uuid.uuid4()),
                name=instance.metadata.name,
                status=TraceSpanStatus.IN_PROGRESS,
                description=instance.metadata.description,
                children=[],
                trace_uuid=self.active_trace_uuid,
                parent_uuid=base_span.uuid,
                start_time=perf_counter(),
                input=bound_args.arguments,
            )

            trace_manager.add_span(tool_span)
            trace_manager.add_span_to_trace(tool_span)

            # adding this tool span id to the metadata of the parent span
            trace_manager.get_span_by_uuid(base_span.uuid).metadata = {
                "tool_span_id": tool_span.uuid
            }

        return base_span

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[LlamaIndexBaseSpan]:
        base_span = trace_manager.get_span_by_uuid(id_)

        if base_span is None:
            return None

        if isinstance(instance, AsyncBaseTool):
            tool_span_id = base_span.metadata.get("tool_span_id")
            if tool_span_id:
                tool_span = trace_manager.get_span_by_uuid(tool_span_id)
                if tool_span:
                    tool_span.end_time = perf_counter()
                    tool_span.status = TraceSpanStatus.SUCCESS
                    tool_span.output = result
                    trace_manager.remove_span(tool_span.uuid)

        base_span.end_time = perf_counter()
        base_span.status = TraceSpanStatus.SUCCESS
        base_span.output = result
        trace_manager.remove_span(base_span.uuid)

        if base_span.parent_uuid is None:
            trace_manager.end_trace(base_span.trace_uuid)
            self.active_trace_uuid = None

        return base_span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[LlamaIndexBaseSpan]:
        base_span = trace_manager.get_span_by_uuid(id_)
        if base_span is None:
            return None

        base_span.end_time = perf_counter()
        base_span.status = (
            TraceSpanStatus.SUCCESS
        )  # find a way to add error and handle the span without the parent id

        if base_span.parent_uuid is None:
            trace_manager.end_trace(base_span.trace_uuid)
            self.active_trace_uuid = None

        return base_span


def instrument_llama_index(dispatcher: Dispatcher):
    handler = LLamaIndexHandler()
    dispatcher.add_event_handler(handler)
    dispatcher.add_span_handler(handler)
    return None
