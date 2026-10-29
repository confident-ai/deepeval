"""deepeval's AWS Bedrock AgentCore integration, built on confident-trace.

confident-trace traces each AgentCore invocation once deepeval starts it with
the ``agentcore`` instrumentation, and the Strands agent serving it with the
``strands`` one.
"""

from typing import List, Optional

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.adapters.strands import SCOPE
from deepeval.tracing.otel.setup import (
    ensure_confident_trace,
    set_trace_defaults,
)


def instrument_agentcore(
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
    """Trace AWS Bedrock AgentCore and its Strands agents for deepeval.

    The kwargs are defaults for every trace; ``update_current_trace(...)``
    overrides them at runtime. Span fields are set at the call site with
    ``update_current_span(...)`` or ``next_*_span(...)``. ``api_key`` is only
    needed to send traces to Confident AI under a key other than
    ``CONFIDENT_API_KEY``.
    """
    with capture_tracing_integration(Integration.AGENTCORE):
        ensure_confident_trace(("agentcore", "strands"), api_key=api_key)
        set_trace_defaults(SCOPE, dict(locals()))
