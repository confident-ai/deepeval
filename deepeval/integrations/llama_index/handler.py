from typing import Any, Dict, Optional
import inspect
from time import perf_counter
import uuid

from llama_index.core.agent.workflow.workflow_events import (
    AgentWorkflowStartEvent,
)
from deepeval.integrations.llama_index.utils import (
    extract_output_from_llm_chat_end_event,
)
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.types import (
    ToolSpan,
    AgentSpan,
    BaseSpan,
    LlmSpan,
    TraceSpanStatus,
)
from deepeval.tracing.trace_context import (
    current_llm_context,
    current_agent_context,
)
from deepeval.test_case import ToolCall
from deepeval.tracing.utils import make_json_serializable

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
    from llama_index_instrumentation.dispatcher import Dispatcher
    from deepeval.integrations.llama_index.utils import (
        parse_id,
        prepare_input_llm_test_case_params,
        prepare_output_llm_test_case_params,
    )

    llama_index_installed = True
except:
    llama_index_installed = False


def is_llama_index_installed():
    if not llama_index_installed:
        raise ImportError(
            "llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice."
        )


class LLamaIndexHandler(BaseEventHandler, BaseSpanHandler):
    root_span_trace_id_map: Dict[str, str] = {}
    open_ai_astream_to_llm_span_map: Dict[str, str] = {}

    def __init__(self):
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

            llm_span_context = current_llm_context.get()
            # create the span
            llm_span = LlmSpan(
                name="ConfidentLLMSpan",
                uuid=str(uuid.uuid4()),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=trace_manager.get_span_by_uuid(
                    event.span_id
                ).trace_uuid,
                parent_uuid=event.span_id,
                start_time=perf_counter(),
                model=getattr(event, "model_dict", {}).get(
                    "model", "unknown"
                ),  # check the model name not coming in this option
                input=input_messages,
                output="",
                metrics=llm_span_context.metrics if llm_span_context else None,
                metric_collection=(
                    llm_span_context.metric_collection
                    if llm_span_context
                    else None
                ),
                prompt=llm_span_context.prompt if llm_span_context else None,
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
                    llm_span.input = llm_span.input
                    llm_span.output = extract_output_from_llm_chat_end_event(
                        event
                    )
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
        class_name, method_name = parse_id(id_)

        # check if it is a root span
        if parent_span_id is None:
            trace_uuid = trace_manager.start_new_trace().uuid
        elif class_name == "Workflow" and method_name == "run":
            trace_uuid = trace_manager.start_new_trace().uuid
            parent_span_id = None  # since workflow is the root span, we need to set the parent span id to None
        elif trace_manager.get_span_by_uuid(parent_span_id):
            trace_uuid = trace_manager.get_span_by_uuid(
                parent_span_id
            ).trace_uuid
        else:
            trace_uuid = trace_manager.start_new_trace().uuid

        self.root_span_trace_id_map[id_] = trace_uuid

        # default span
        span = BaseSpan(
            uuid=id_,
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=trace_uuid,
            parent_uuid=parent_span_id,
            start_time=perf_counter(),
            name=method_name if method_name else instance.__class__.__name__,
            input=bound_args.arguments,
        )

        # conditions to qualify as agent start run span
        if method_name == "run":
            agent_span_context = current_agent_context.get()
            start_event = bound_args.arguments.get("start_event")

            if start_event and isinstance(start_event, AgentWorkflowStartEvent):
                input = start_event.model_dump()

            else:
                input = bound_args.arguments

            span = AgentSpan(
                uuid=id_,
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=trace_uuid,
                parent_uuid=parent_span_id,
                start_time=perf_counter(),
                name="Agent",  # TODO: decide the name of the span
                input=input,
                metrics=(
                    agent_span_context.metrics if agent_span_context else None
                ),
                metric_collection=(
                    agent_span_context.metric_collection
                    if agent_span_context
                    else None
                ),
            )
        elif method_name == "acall":
            span = ToolSpan(
                uuid=id_,
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=trace_uuid,
                parent_uuid=parent_span_id,
                start_time=perf_counter(),
                input=bound_args.arguments,
                name="Tool",
            )
        # prepare input test case params for the span
        prepare_input_llm_test_case_params(
            class_name, method_name, span, bound_args.arguments
        )
        trace_manager.add_span(span)
        trace_manager.add_span_to_trace(span)

        return span

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

        class_name, method_name = parse_id(id_)
        if method_name == "call_tool":
            output_json = make_json_serializable(result)
            if output_json and isinstance(output_json, dict):
                if base_span.tools_called is None:
                    base_span.tools_called = []
                base_span.tools_called.append(
                    ToolCall(
                        name=output_json.get("tool_name", "Tool"),
                        input_parameters=output_json.get("tool_kwargs", {}),
                        output=output_json.get("tool_output", {}),
                    )
                )
        base_span.end_time = perf_counter()
        base_span.status = TraceSpanStatus.SUCCESS
        base_span.output = result

        if isinstance(base_span, ToolSpan):
            result_json = make_json_serializable(result)
            if result_json and isinstance(result_json, dict):
                base_span.name = result_json.get("tool_name", "Tool")

        if base_span.llm_test_case:
            class_name, method_name = parse_id(id_)
            prepare_output_llm_test_case_params(
                class_name, method_name, result, base_span
            )

        if base_span.metrics:
            trace_manager.integration_traces_to_evaluate.append(
                trace_manager.get_trace_by_uuid(base_span.trace_uuid)
            )

        trace_manager.remove_span(base_span.uuid)

        if base_span.parent_uuid is None:
            trace_manager.end_trace(base_span.trace_uuid)
            self.root_span_trace_id_map.pop(base_span.uuid)

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
            self.root_span_trace_id_map.pop(base_span.uuid)

        return base_span


def instrument_llama_index(dispatcher: Dispatcher):
    with capture_tracing_integration("llama_index"):
        handler = LLamaIndexHandler()
        dispatcher.add_event_handler(handler)
        dispatcher.add_span_handler(handler)
        return None
