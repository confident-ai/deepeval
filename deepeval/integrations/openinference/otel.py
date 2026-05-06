"""``instrument_openinference(...)`` — wire OpenInference spans into deepeval.

Pydantic AI POC pattern: ``OpenInferenceSpanInterceptor`` then
``ContextAwareSpanProcessor`` (REST when a deepeval trace context is
active or evaluating, OTLP otherwise). Idempotent on the same
``TracerProvider`` — subsequent calls mutate settings in place instead of
stacking processors (community OpenInference instrumentors install onto
the global provider, so stacking would corrupt contextvars and leak
settings).

Span-level config (per-call ``metric_collection``, ``metrics``,
``prompt``) belongs on ``with next_*_span(...)`` / ``update_current_span(...)``
— see ``deepeval/integrations/README.md``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.telemetry import capture_tracing_integration

logger = logging.getLogger(__name__)
settings = get_settings()


try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    _opentelemetry_installed = True
except ImportError:
    _opentelemetry_installed = False


# Tracks the (interceptor, casp) pair we attached per provider so repeat
# ``instrument_openinference(...)`` calls mutate settings in place rather
# than stack — see module docstring.
_attached_processors: Dict[int, Tuple[object, object]] = {}


def _require_opentelemetry() -> None:
    if not _opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. "
            "Install it with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )


# Mirrors ``OpenInferenceInstrumentationSettings._REMOVED_KWARGS`` for error
# reporting.
_REMOVED_INSTRUMENT_KWARGS = (
    "is_test_mode",
    "agent_metric_collection",
    "llm_metric_collection",
    "tool_metric_collection_map",
    "trace_metric_collection",
    "agent_metrics",
    "confident_prompt",
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
    **removed_kwargs,
) -> None:
    """Attach Confident AI / deepeval telemetry to any OpenInference instrumentor.

    All kwargs are optional and trace-level; span-level fields go on
    ``with next_*_span(...)`` / ``update_current_span(...)``. Routing is
    REST when a deepeval trace context is active (``@observe`` /
    ``with trace(...)``) or ``trace_manager.is_evaluating`` is True;
    OTLP otherwise.
    """
    if removed_kwargs:
        offending = ", ".join(sorted(removed_kwargs))
        raise TypeError(
            f"instrument_openinference: unexpected keyword argument(s) "
            f"{offending}. Span-level kwargs were removed in the OTel POC "
            "migration; use ``with next_*_span(...)`` or "
            "``update_current_span(...)``. "
            "See deepeval/integrations/README.md."
        )

    with capture_tracing_integration("openinference"):
        _require_opentelemetry()

        if not api_key:
            api_key = get_confident_api_key()

        # Deferred so ``_require_opentelemetry`` fails cleanly when OTel is missing.
        from deepeval.tracing.otel.context_aware_processor import (
            ContextAwareSpanProcessor,
        )

        from .instrumentator import (
            OpenInferenceInstrumentationSettings,
            OpenInferenceSpanInterceptor,
        )

        openinference_settings = OpenInferenceInstrumentationSettings(
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
        )

        # Reuse the active TracerProvider; create + set globally if it's a no-op.
        current_provider = trace.get_tracer_provider()
        if type(current_provider).__name__ in (
            "ProxyTracerProvider",
            "NoOpTracerProvider",
        ):
            tracer_provider = TracerProvider()
            try:
                trace.set_tracer_provider(tracer_provider)
                logger.debug("Created and registered a new TracerProvider.")
            except Exception as exc:
                logger.warning("Could not set global tracer provider: %s", exc)
            current_provider = trace.get_tracer_provider()

        if not hasattr(current_provider, "add_span_processor"):
            logger.warning(
                "The active TracerProvider (%s) does not support "
                "add_span_processor. OpenInference telemetry cannot be attached.",
                type(current_provider).__name__,
            )
            return

        existing = _attached_processors.get(id(current_provider))
        if existing is not None:
            # Mutate settings in place so repeat calls fully replace prior
            # trace-level config without layering another processor.
            interceptor, _casp = existing
            interceptor.settings = openinference_settings
            logger.debug(
                "OpenInference telemetry re-configured (env=%s).",
                openinference_settings.environment,
            )
            return

        # Registration order matters: interceptor writes ``confident.*`` attrs
        # before CASP routes the span (OTel runs processors in order on on_end).
        interceptor = OpenInferenceSpanInterceptor(openinference_settings)
        casp = ContextAwareSpanProcessor(api_key=api_key)
        current_provider.add_span_processor(interceptor)
        current_provider.add_span_processor(casp)
        _attached_processors[id(current_provider)] = (interceptor, casp)

        logger.info(
            "Confident AI OpenInference telemetry attached (env=%s).",
            openinference_settings.environment,
        )
