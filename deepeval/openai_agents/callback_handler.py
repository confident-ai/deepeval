from time import perf_counter

from deepeval.tracing.tracing import (
    Observer,
    current_span_context,
    trace_manager,
)
from deepeval.openai_agents.extractors import (
    update_span_properties,
    update_trace_properties_from_span_data,
)
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.utils import make_json_serializable
from deepeval.tracing.types import (
    BaseSpan,
    LlmSpan,
    TraceSpanStatus,
)

try:
    from agents.tracing import Span, Trace, TracingProcessor
    from agents.tracing.span_data import (
        AgentSpanData,
        CustomSpanData,
        FunctionSpanData,
        GenerationSpanData,
        GuardrailSpanData,
        HandoffSpanData,
        MCPListToolsSpanData,
        ResponseSpanData,
        SpanData,
    )
    from deepeval.openai_agents.patch import (
        patch_default_agent_run_single_turn,
        patch_default_agent_run_single_turn_streamed,
    )

    openai_agents_available = True
except ImportError:
    openai_agents_available = False


def _check_openai_agents_available():
    if not openai_agents_available:
        raise ImportError(
            "openai-agents is required for this integration. Install it via your package manager"
        )


class DeepEvalTracingProcessor(TracingProcessor):
    def __init__(self) -> None:
        _check_openai_agents_available()
        patch_default_agent_run_single_turn()
        patch_default_agent_run_single_turn_streamed()
        self.span_observers: dict[str, Observer] = {}

    def on_trace_start(self, trace: "Trace") -> None:
        trace_dict = trace.export()
        _trace_uuid = trace_dict.get("id")
        _thread_id = trace_dict.get("group_id")
        _trace_name = trace_dict.get("workflow_name")
        _trace_metadata = trace_dict.get("metadata")

        _trace = trace_manager.start_new_trace(trace_uuid=str(_trace_uuid))
        _trace.thread_id = str(_thread_id)
        _trace.name = str(_trace_name)
        _trace.metadata = make_json_serializable(_trace_metadata)
        current_trace_context.set(_trace)

        trace_manager.add_span(  # adds a dummy root span
            BaseSpan(
                uuid=_trace_uuid,
                trace_uuid=_trace_uuid,
                parent_uuid=None,
                start_time=perf_counter(),
                name=_trace_name,
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
            )
        )

    def on_trace_end(self, trace: "Trace") -> None:
        trace_dict = trace.export()
        _trace_uuid = trace_dict.get("id")
        _trace_name = trace_dict.get("workflow_name")

        trace_manager.remove_span(_trace_uuid)  # removing the dummy root span
        trace_manager.end_trace(_trace_uuid)
        current_trace_context.set(None)

    def on_span_start(self, span: "Span") -> None:
        if not span.started_at:
            return
        current_span = current_span_context.get()
        if current_span and isinstance(
            current_span, LlmSpan
        ):  # llm span started by
            return

        span_type = self.get_span_kind(span.span_data)
        observer = Observer(span_type=span_type, func_name="NA")
        if span_type == "llm":
            observer.observe_kwargs["model"] = "temporary model"
        observer.update_span_properties = (
            lambda span_type: update_span_properties(span_type, span.span_data)
        )
        self.span_observers[span.span_id] = observer
        observer.__enter__()

    def on_span_end(self, span: "Span") -> None:
        update_trace_properties_from_span_data(
            current_trace_context.get(), span.span_data
        )

        span_type = self.get_span_kind(span.span_data)
        current_span = current_span_context.get()
        if (
            current_span
            and isinstance(current_span, LlmSpan)
            and span_type == "llm"
        ):  # addtional check if the span kind data is llm too
            update_span_properties(current_span, span.span_data)

        observer = self.span_observers.pop(span.span_id, None)
        if observer:
            observer.__exit__(None, None, None)

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_span_kind(self, span_data: "SpanData") -> str:
        if isinstance(span_data, AgentSpanData):
            return "agent"
        if isinstance(span_data, FunctionSpanData):
            return "tool"
        if isinstance(span_data, MCPListToolsSpanData):
            return "tool"
        if isinstance(span_data, GenerationSpanData):
            return "llm"
        if isinstance(span_data, ResponseSpanData):
            return "llm"
        if isinstance(span_data, HandoffSpanData):
            return "custom"
        if isinstance(span_data, CustomSpanData):
            return "base"
        if isinstance(span_data, GuardrailSpanData):
            return "base"
        return "base"
