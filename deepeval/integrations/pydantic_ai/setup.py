from typing import Optional
import deepeval
from deepeval.telemetry import capture_tracing_integration
from deepeval.confident.api import get_confident_api_key

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


OTLP_ENDPOINT = "https://otel.confident-ai.com/v1/traces"


def instrument_pydantic_ai(api_key: Optional[str] = None):
    with capture_tracing_integration("pydantic_ai"):
        is_opentelemetry_available()

        if api_key:
            deepeval.login(api_key)

        api_key = get_confident_api_key()

        if not api_key:
            raise ValueError("No api key provided.")

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
        trace.set_tracer_provider(tracer_provider)

        # create an instrumented exporter
        from pydantic_ai.models.instrumented import InstrumentationSettings
        from pydantic_ai import Agent

        instrumentation_settings = InstrumentationSettings(
            tracer_provider=tracer_provider
        )

        # instrument all agents
        Agent.instrument_all(instrument=instrumentation_settings)
