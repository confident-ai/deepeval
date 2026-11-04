"""Wire Microsoft Agent Framework OpenTelemetry spans into DeepEval."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from deepeval.confident.api import get_confident_api_key
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    _opentelemetry_installed = True
except ImportError:
    _opentelemetry_installed = False


_attached_processors: Dict[int, Tuple[object, object]] = {}


def _require_dependencies() -> None:
    if not _opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. Install it with: "
            "pip install opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http"
        )
    try:
        import agent_framework.observability  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Microsoft Agent Framework is not available. "
            "Install it with: pip install agent-framework"
        ) from exc


def instrument_microsoft_agent_framework(
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
    capture_content: bool = False,
) -> None:
    """Attach DeepEval telemetry to Microsoft Agent Framework.

    Agent Framework instrumentation is enabled through its public
    observability API. Prompt, response, and tool content remains disabled
    unless ``capture_content=True`` is explicitly requested.
    """
    with capture_tracing_integration(Integration.MICROSOFT_AGENT_FRAMEWORK):
        _require_dependencies()

        if not api_key:
            api_key = get_confident_api_key()

        from agent_framework.observability import enable_instrumentation

        from deepeval.tracing.otel.context_aware_processor import (
            ContextAwareSpanProcessor,
        )

        from .instrumentator import (
            MicrosoftAgentFrameworkInstrumentationSettings,
            MicrosoftAgentFrameworkSpanInterceptor,
        )

        framework_settings = MicrosoftAgentFrameworkInstrumentationSettings(
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
                "add_span_processor. Microsoft Agent Framework telemetry "
                "cannot be attached.",
                type(current_provider).__name__,
            )
            return

        existing = _attached_processors.get(id(current_provider))
        if existing is not None:
            interceptor, _casp = existing
            interceptor.settings = framework_settings
        else:
            interceptor = MicrosoftAgentFrameworkSpanInterceptor(
                framework_settings
            )
            casp = ContextAwareSpanProcessor(api_key=api_key)
            current_provider.add_span_processor(interceptor)
            current_provider.add_span_processor(casp)
            _attached_processors[id(current_provider)] = (interceptor, casp)

        # Agent Framework instrumentation is enabled by default, but calling
        # the public API makes the integration deterministic. This intentionally
        # does not call configure_otel_providers(): DeepEval owns the provider.
        enable_instrumentation(enable_sensitive_data=capture_content)

        logger.info(
            "Confident AI Microsoft Agent Framework telemetry attached "
            "(env=%s, capture_content=%s).",
            framework_settings.environment,
            capture_content,
        )
