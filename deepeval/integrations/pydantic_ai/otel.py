import warnings
from typing import Optional
from deepeval.telemetry import capture_tracing_integration
from deepeval.config.settings import get_settings
import logging

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    opentelemetry_installed = True
except:
    opentelemetry_installed = False


def is_opentelemetry_available():
    if not opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. Please install it with `pip install opentelemetry-sdk`."
        )
    return True


logger = logging.getLogger(__name__)
settings = get_settings()

settings = get_settings()
# OTLP_ENDPOINT = "https://otel.confident-ai.com/v1/traces"

OTLP_ENDPOINT = str(settings.CONFIDENT_OTEL_URL) + "v1/traces"


def instrument_pydantic_ai(api_key: Optional[str] = None):
    warnings.warn(
        "instrument_pydantic_ai is deprecated and will be removed in a future version. "
        "Please use the new ConfidentInstrumentationSettings instead. Docs: https://www.confident-ai.com/docs/integrations/third-party/pydantic-ai",
        DeprecationWarning,
        stacklevel=2,
    )

    with capture_tracing_integration("pydantic_ai"):
        is_opentelemetry_available()

        # create a new tracer provider
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=OTLP_ENDPOINT,
                    headers={"x-confident-api-key": api_key},
                )
            )
        )
        try:
            trace.set_tracer_provider(tracer_provider)
        except Exception as e:
            # Handle case where provider is already set (optional warning)
            logger.warning(f"Could not set global tracer provider: {e}")

        # create an instrumented exporter
        from pydantic_ai.models.instrumented import InstrumentationSettings
        from pydantic_ai import Agent

        instrumentation_settings = InstrumentationSettings(
            tracer_provider=tracer_provider
        )

        # instrument all agents
        Agent.instrument_all(instrument=instrumentation_settings)
