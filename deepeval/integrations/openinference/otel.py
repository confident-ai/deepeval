"""Collect OpenInference instrumentors' spans for deepeval.

Any ``openinference-instrumentation-*`` instrumentor traces on the global
tracer provider, which confident-trace runs once deepeval starts it.
"""

from typing import List, Optional

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.adapters.openinference import SCOPE
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.setup import (
    ensure_confident_trace,
    set_trace_defaults,
)


def instrument_openinference(
    api_key: Optional[str] = None,
    name: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    environment: Optional[str] = None,
    metric_collection: Optional[str] = None,
    test_case_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    integration: Optional[str] = None,
) -> None:
    """Trace whatever OpenInference instrumentors run for deepeval.

    Pair it with an instrumentor, e.g. ``OpenAIInstrumentor().instrument()``.
    The kwargs are defaults for every trace; ``update_current_trace(...)``
    overrides them at runtime. Span fields are set at the call site with
    ``update_current_span(...)`` or ``next_*_span(...)``. ``integration``
    labels the spans. ``api_key`` is only needed to send traces to Confident
    AI under a key other than ``CONFIDENT_API_KEY``.
    """
    with capture_tracing_integration(Integration.OPEN_INFERENCE):
        ensure_confident_trace(api_key=api_key)
        label = (
            {ConfidentAttr.SPAN_INTEGRATION: integration} if integration else {}
        )
        set_trace_defaults(SCOPE, dict(locals()), **label)
