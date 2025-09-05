from deepeval.tracing.tracing import (
    Observer,
    current_span_context,
)
from deepeval.openai_agents.extractors import *
from deepeval.tracing.context import current_trace_context

try:
    from agents.tracing import Span, Trace, TracingProcessor
    from agents.tracing.span_data import (
        AgentSpanData,
        CustomSpanData,
        FunctionSpanData,
        GenerationSpanData,
        GuardrailSpanData,
        HandoffSpanData,
        ResponseSpanData,
        SpanData,
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
        self.root_span_observers: dict[str, Observer] = {}
        self.span_observers: dict[str, Observer] = {}

    def on_trace_start(self, trace: "Trace") -> None:
        pass

    def on_trace_end(self, trace: "Trace") -> None:
        pass

    def on_span_start(self, span: "Span") -> None:
        if not span.started_at:
            return
        span_type = self.get_span_kind(span.span_data)
        if span_type == "agent":
            if isinstance(span.span_data, AgentSpanData):
                current_trace = current_trace_context.get()
                if current_trace:
                    current_trace.name = span.span_data.name

        if span_type == "tool":
            return
        elif span_type == "llm":
            return
        else:
            observer = Observer(span_type=span_type, func_name="NA")
            observer.update_span_properties = (
                lambda base_span: update_span_properties(
                    base_span, span.span_data
                )
            )
            self.span_observers[span.span_id] = observer
            observer.__enter__()

    def on_span_end(self, span: "Span") -> None:
        span_type = self.get_span_kind(span.span_data)
        if span_type == "llm":
            current_span = current_span_context.get()
            if current_span:
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
