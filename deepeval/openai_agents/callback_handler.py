from deepeval.tracing.tracing import (
    Observer, 
    SpanType, 
)
from deepeval.openai_agents.extractors import *

# check openai agents availability
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


class OpenAIAgentsCallbackHandler(TracingProcessor):
    def __init__(self) -> None:
        _check_openai_agents_available()
        self.root_span_observers: dict[str, Observer] = {}
        self.span_observers: dict[str, Observer] = {}
        
    def on_trace_start(self, trace: Trace) -> None:
        observer = Observer(span_type=SpanType.AGENT, func_name=trace.name)
        self.root_span_observers[trace.trace_id] = observer
        observer.__enter__()

    def on_trace_end(self, trace: Trace) -> None:
        observer = self.root_span_observers.pop(trace.trace_id, None)
        if observer:
            observer.__exit__(None, None, None)

    def on_span_start(self, span: Span) -> None:
        if not span.started_at:
            return
        span_name = self.get_span_name(span)
        span_type = self.get_span_kind(span.span_data)
        observer = Observer(span_type=span_type, func_name=span_name)
        if span_type == "llm":
            observer.observe_kwargs["model"] = "temporary model"
        observer.custom_update_span_attributes = (
            lambda span_type: custom_update_span_attributes(span_type, span.span_data)
        )
        self.span_observers[span.span_id] = observer
        observer.__enter__()

    def on_span_end(self, span: Span) -> None:
        observer = self.span_observers.pop(span.span_id, None)
        if observer:
            observer.__exit__(None, None, None)

    def force_flush(self) -> None:
        """Forces an immediate flush of all queued spans/traces."""
        # TODO
        pass

    def shutdown(self) -> None:
        """Called when the application stops."""
        # TODO
        pass

    def get_span_name(self, span_data: SpanData) -> str:
        if hasattr(span_data, "name") and isinstance(span_data.name, str):
            return span_data.name
        if isinstance(span_data, HandoffSpanData) and span_data.to_agent:
            return f"handoff to {span_data.to_agent}"
        return "NA"

    def get_span_kind(self, span_data: SpanData) -> str:
        if isinstance(span_data, AgentSpanData):
            return "agent"
        if isinstance(span_data, FunctionSpanData):
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