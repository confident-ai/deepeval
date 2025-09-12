import os
from typing import Optional
from opentelemetry import baggage
from opentelemetry.trace import Tracer as OTelTracer
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    is_opentelemetry_installed = True
except Exception:
    is_opentelemetry_installed = False


def is_opentelemetry_available():
    if not is_opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. Please install it with `pip install opentelemetry-exporter-otlp-proto-http`."
        )
    return True


from deepeval.confident.api import get_confident_api_key

OTLP_ENDPOINT = (
    os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    else "https://otel.confident-ai.com"
)
# OTLP_ENDPOINT = "http://127.0.0.1:4318"

# Module-level globals to be imported and used by other code
GLOBAL_TEST_RUN_TRACER_PROVIDER: Optional[TracerProvider] = None
GLOBAL_TEST_RUN_TRACER: Optional[OTelTracer] = None


class RunIdSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context):
        run_id = baggage.get_baggage(
            "confident.test_run.id", context=parent_context
        )
        if run_id:
            span.set_attribute("confident.test_run.id", run_id)

    def on_end(self, span) -> None:  # type: ignore[override]
        # No-op
        return None

    def shutdown(self) -> None:  # type: ignore[override]
        # No-op
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        # No-op
        return True


def init_global_test_run_tracer(api_key: Optional[str] = None):
    is_opentelemetry_available()
    api_key = get_confident_api_key()
    if api_key is None:
        raise ValueError("CONFIDENT_API_KEY is not set")

    provider = TracerProvider()
    exporter = OTLPSpanExporter(
        endpoint=f"{OTLP_ENDPOINT}/v1/traces",
        headers={"x-confident-api-key": api_key},
    )
    provider.add_span_processor(RunIdSpanProcessor())
    provider.add_span_processor(BatchSpanProcessor(span_exporter=exporter))
    tracer = provider.get_tracer("deepeval_tracer")

    global GLOBAL_TEST_RUN_TRACER_PROVIDER
    global GLOBAL_TEST_RUN_TRACER
    GLOBAL_TEST_RUN_TRACER_PROVIDER = provider
    GLOBAL_TEST_RUN_TRACER = tracer

    return provider, tracer
