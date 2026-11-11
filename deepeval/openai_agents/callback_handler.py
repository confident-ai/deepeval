"""deepeval's OpenAI Agents integration, built on confident-trace.

confident-trace traces agent runs through OpenInference's processor once
deepeval starts it with the ``openai_agents`` instrumentation.
``DeepEvalTracingProcessor`` adds what only deepeval knows: each run's trace
settings, and the settings a deepeval ``Agent`` gives its spans.
"""

from typing import Any, Dict

from opentelemetry import trace as otel_trace

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.bridge import trace_field_attributes
from deepeval.tracing.otel.setup import ensure_confident_trace

try:
    from agents.tracing import Span, Trace, TracingProcessor
    from agents.tracing.span_data import AgentSpanData

    from deepeval.openai_agents.patch import (
        agent_spans,
        patch_run_steps,
    )

    openai_agents_available = True
except ImportError:
    openai_agents_available = False
    TracingProcessor = object

INSTRUMENTATIONS = ("openai_agents", "openai")


def _check_openai_agents_available():
    if not openai_agents_available:
        raise ImportError(
            "openai-agents is required for this integration. Install it via your package manager"
        )
    try:
        import openinference.instrumentation.openai_agents  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "openinference-instrumentation-openai-agents is required to trace "
            "OpenAI Agents. Install it with "
            "`pip install openinference-instrumentation-openai-agents`."
        ) from error


class DeepEvalTracingProcessor(TracingProcessor):
    """Add it with ``agents.add_trace_processor(DeepEvalTracingProcessor())``.

    It runs after OpenInference's processor, so the OpenTelemetry span for
    each SDK span is current when that span starts.
    """

    def __init__(self) -> None:
        _check_openai_agents_available()
        with capture_tracing_integration(Integration.OPENAI_AGENTS):
            ensure_confident_trace(INSTRUMENTATIONS)
        patch_run_steps()
        self._trace_fields: Dict[str, Dict[str, Any]] = {}

    def on_trace_start(self, trace: "Trace") -> None:
        exported = trace.export() or {}
        self._trace_fields[trace.trace_id] = trace_field_attributes(
            {
                "name": exported.get("workflow_name"),
                "thread_id": exported.get("group_id"),
                "metadata": exported.get("metadata"),
            }
        )

    def on_trace_end(self, trace: "Trace") -> None:
        self._trace_fields.pop(trace.trace_id, None)

    def on_span_start(self, span: "Span") -> None:
        otel_span = otel_trace.get_current_span()
        if not otel_span.get_span_context().is_valid:
            return
        if span.parent_id is None:
            # A run's first spans carry its trace settings.
            otel_span.set_attributes(self._trace_fields.get(span.trace_id, {}))
        if isinstance(span.span_data, AgentSpanData):
            agent_spans[span.span_id] = otel_span

    def on_span_end(self, span: "Span") -> None:
        agent_spans.pop(span.span_id, None)

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
