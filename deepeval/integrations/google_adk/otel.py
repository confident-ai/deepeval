"""deepeval's Google ADK integration, built on confident-trace.

confident-trace reads the telemetry ADK emits once deepeval starts it with the
``google_adk`` instrumentation.
"""

from typing import List, Optional

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.adapters.google_adk import SCOPE
from deepeval.tracing.otel.setup import (
    ensure_confident_trace,
    set_trace_defaults,
)


def instrument_google_adk(
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
) -> None:
    """Trace Google ADK agents for deepeval.

    The kwargs are defaults for every trace; ``update_current_trace(...)``
    overrides them at runtime. Span fields are set at the call site with
    ``update_current_span(...)`` or ``next_*_span(...)``. ``api_key`` is only
    needed to send traces to Confident AI under a key other than
    ``CONFIDENT_API_KEY``.
    """
    with capture_tracing_integration(Integration.GOOGLE_ADK):
        ensure_confident_trace(("google_adk",), api_key=api_key)
        set_trace_defaults(SCOPE, dict(locals()))
