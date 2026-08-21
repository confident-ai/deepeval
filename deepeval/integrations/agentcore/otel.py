"""``instrument_agentcore(...)`` — wire AgentCore spans into deepeval.

Pydantic AI POC pattern: ``AgentCoreSpanInterceptor`` then
``ContextAwareSpanProcessor`` (REST when a deepeval trace context is
active or evaluating, OTLP otherwise). Idempotent on the same
``TracerProvider`` — subsequent calls mutate settings in place instead
of stacking processors (Strands writes to the global provider, so
stacking would corrupt contextvars and leak settings). See
``deepeval.integrations.otel_instrumentation.base_instrumentation``
for the attach logic.

Span-level config (per-call ``metric_collection``, ``metrics``,
``prompt``) belongs on ``with next_*_span(...)`` / ``update_current_span(...)``
— see ``deepeval/integrations/README.md``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from deepeval.confident.api import get_confident_api_key
from deepeval.integrations.otel_instrumentation.base_instrumentation import (
    attach_span_interceptor,
    raise_on_removed_kwargs,
    require_opentelemetry,
)
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration

logger = logging.getLogger(__name__)


# Tracks the (interceptor, casp) pair we attached per provider so repeat
# ``instrument_agentcore(...)`` calls mutate settings in place rather than
# stack — see module docstring.
_attached_processors: Dict[int, Tuple[object, object]] = {}


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
    **removed_kwargs,
) -> None:
    """Attach Confident AI / deepeval telemetry to AWS Bedrock AgentCore.

    All kwargs are optional and trace-level; span-level fields go on
    ``with next_*_span(...)`` / ``update_current_span(...)``. Routing is
    REST when a deepeval trace context is active (``@observe`` /
    ``with trace(...)``) or ``trace_manager.is_evaluating`` is True;
    OTLP otherwise.
    """
    raise_on_removed_kwargs("instrument_agentcore", removed_kwargs)

    with capture_tracing_integration(Integration.AGENTCORE):
        require_opentelemetry()

        if not api_key:
            api_key = get_confident_api_key()

        # Deferred so ``require_opentelemetry`` fails cleanly first.
        from .instrumentator import (
            AgentCoreInstrumentationSettings,
            AgentCoreSpanInterceptor,
        )

        attach_span_interceptor(
            interceptor_cls=AgentCoreSpanInterceptor,
            instrumentation_settings=AgentCoreInstrumentationSettings(
                api_key=api_key,
                name=name,
                thread_id=thread_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                environment=environment,
                metric_collection=metric_collection,
                test_case_id=test_case_id,
                turn_id=turn_id,
            ),
            api_key=api_key,
            registry=_attached_processors,
            label="AgentCore",
        )
