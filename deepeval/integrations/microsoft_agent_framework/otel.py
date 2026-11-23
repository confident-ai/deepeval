"""Send Microsoft Agent Framework's native GenAI spans to confident-trace."""

from typing import List, Optional

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.setup import (
    ensure_confident_trace,
    set_trace_defaults,
)

_SCOPE = "agent_framework"


def instrument_microsoft_agent_framework(
    name: Optional[str] = None,
    thread_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    environment: Optional[str] = None,
    capture_content: bool = False,
) -> None:
    """Trace Agent Framework through its public observability API.

    Each trace uses the supplied defaults unless overridden at runtime.
    Set ``capture_content=True`` to include messages and tool content.
    """
    trace_defaults = dict(locals())
    try:
        from agent_framework.observability import enable_instrumentation
    except ImportError as error:
        raise ImportError(
            "Install Agent Framework with `pip install agent-framework`."
        ) from error

    integration = Integration.MICROSOFT_AGENT_FRAMEWORK
    with capture_tracing_integration(integration):
        if ensure_confident_trace(global_provider=True):
            set_trace_defaults(
                _SCOPE,
                trace_defaults,
                **{ConfidentAttr.SPAN_INTEGRATION: integration.value},
            )
            enable_instrumentation(enable_sensitive_data=capture_content)
