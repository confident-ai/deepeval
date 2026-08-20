"""Setup path shared by the OTel integrations' ``instrument_*()`` entry points.

Two halves of wiring an integration up: ``BaseInstrumentationSettings`` holds
the trace-level defaults a caller passes to ``instrument_*()``, and
``attach_span_interceptor`` registers that integration's ``SpanInterceptor``
plus ``ContextAwareSpanProcessor`` against the global ``TracerProvider``.

The interceptor's own runtime helpers — placeholder lifecycle, trace-attr
resolution, the OTel SDK import shim — live in the sibling ``utils`` module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from deepeval.config.settings import get_settings
from deepeval.integrations.otel_instrumentation.utils import (
    OTEL_INSTALL_HINT,
    dependency_installed,
    is_dependency_installed,
)
from deepeval.tracing.tracing import trace_manager

logger = logging.getLogger(__name__)
settings = get_settings()


def raise_on_removed_kwargs(
    caller: str, removed_kwargs: Dict[str, Any]
) -> None:
    """Turn span-level kwargs removed in the OTel migration into a crisp error.

    Callers accept ``**removed_kwargs`` purely so the failure names the
    offending arguments instead of surfacing a bare ``TypeError``.
    """
    if not removed_kwargs:
        return
    offending = ", ".join(sorted(removed_kwargs))
    raise TypeError(
        f"{caller}: unexpected keyword argument(s) {offending}. Span-level "
        "kwargs were removed in the OTel POC migration; use "
        "``with next_*_span(...)`` or ``update_current_span(...)``. "
        "See deepeval/integrations/README.md."
    )


class BaseInstrumentationSettings:
    """Trace-level defaults shared by the OTel instrumentation settings.

    All kwargs are optional. Trace fields are resolved at every span's
    ``on_end`` so runtime ``update_current_trace(...)`` mutations win.
    ``api_key`` is optional; when omitted, the OTel pipeline runs locally but
    the Confident AI backend rejects uploads.

    Subclasses set ``DEFAULT_INTEGRATION`` to their ``Integration`` value.
    """

    # Span-level kwargs removed in the OTel POC migration — raise on use.
    _REMOVED_KWARGS = (
        "is_test_mode",
        "agent_metric_collection",
        "llm_metric_collection",
        "tool_metric_collection_map",
        "trace_metric_collection",
        "agent_metrics",
        "confident_prompt",
    )

    DEFAULT_INTEGRATION: Optional[str] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        metric_collection: Optional[str] = None,
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        environment: Optional[str] = None,
        integration: Optional[str] = None,
        **removed_kwargs: Any,
    ):
        is_dependency_installed()

        # ``**removed_kwargs`` exists only to produce a crisp migration error.
        raise_on_removed_kwargs(type(self).__name__, removed_kwargs)

        if trace_manager.environment is not None:
            _env = trace_manager.environment
        elif environment is not None:
            _env = environment
        elif settings.CONFIDENT_TRACE_ENVIRONMENT is not None:
            _env = settings.CONFIDENT_TRACE_ENVIRONMENT
        else:
            _env = "development"

        if _env not in ("production", "staging", "development", "testing"):
            _env = "development"
        self.environment = _env

        self.api_key = api_key
        self.name = name
        self.thread_id = thread_id
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.metric_collection = metric_collection
        self.test_case_id = test_case_id
        self.turn_id = turn_id
        self.integration = integration or self.DEFAULT_INTEGRATION


# TracerProvider wiring, shared by the ``instrument_*(...)`` entry points.


def require_opentelemetry() -> None:
    if not dependency_installed:
        raise ImportError(
            f"OpenTelemetry SDK is not available. Install it with: "
            f"{OTEL_INSTALL_HINT}"
        )


def attach_span_interceptor(
    *,
    interceptor_cls,
    instrumentation_settings,
    api_key: Optional[str],
    registry: Dict[int, Tuple[object, object]],
    label: str,
) -> None:
    """Register ``interceptor_cls`` + ``ContextAwareSpanProcessor``.

    Reuses the active ``TracerProvider``, creating and globally registering
    one if it's still a no-op. Idempotent per provider: repeat calls mutate
    the existing interceptor's settings in place rather than stacking another
    processor, which would corrupt contextvars and leak settings when the
    framework writes to the global provider.

    ``registry`` is the caller's per-provider bookkeeping dict, kept separate
    per integration so different ``instrument_*()`` calls can coexist.
    ``label`` is the human-readable integration name used in log lines.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    # Deferred so ``require_opentelemetry`` fails cleanly when OTel is missing.
    from deepeval.tracing.otel.context_aware_processor import (
        ContextAwareSpanProcessor,
    )

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
            "add_span_processor. %s telemetry cannot be attached.",
            type(current_provider).__name__,
            label,
        )
        return

    existing = registry.get(id(current_provider))
    if existing is not None:
        # Mutate settings in place so repeat calls fully replace prior
        # trace-level config without layering another processor.
        interceptor, _casp = existing
        interceptor.settings = instrumentation_settings
        logger.debug(
            "%s telemetry re-configured (env=%s).",
            label,
            instrumentation_settings.environment,
        )
        return

    # Registration order matters: the interceptor writes ``confident.*`` attrs
    # before CASP routes the span (OTel runs processors in order on on_end).
    interceptor = interceptor_cls(instrumentation_settings)
    casp = ContextAwareSpanProcessor(api_key=api_key)
    current_provider.add_span_processor(interceptor)
    current_provider.add_span_processor(casp)
    registry[id(current_provider)] = (interceptor, casp)

    logger.info(
        "Confident AI %s telemetry attached (env=%s).",
        label,
        instrumentation_settings.environment,
    )
