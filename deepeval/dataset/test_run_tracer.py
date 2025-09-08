import os
from typing import Optional
from opentelemetry import baggage
from opentelemetry.trace import Tracer as OTelTracer
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)

OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") else "https://otel.confident-ai.com"
# OTLP_ENDPOINT = "http://127.0.0.1:4318"

# Module-level globals to be imported and used by other code
GLOBAL_TEST_RUN_TRACER_PROVIDER: Optional[TracerProvider] = None
GLOBAL_TEST_RUN_TRACER: Optional[OTelTracer] = None

class RunIdSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context):
        run_id = baggage.get_baggage("confident.test.run_id", context=parent_context)
        if run_id:
            span.set_attribute("confident.test.run_id", run_id)
    
    def on_end(self, span) -> None:  # type: ignore[override]
        # No-op
        return None

    def shutdown(self) -> None:  # type: ignore[override]
        # No-op
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        # No-op
        return True

def build_global_test_run_tracer(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("CONFIDENT_API_KEY")
    if api_key is None:
        raise ValueError("CONFIDENT_API_KEY is not set")
    if not OTLP_ENDPOINT:
        raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT is not set")
    
    provider = TracerProvider()
    exporter = OTLPSpanExporter(
        endpoint=f"{OTLP_ENDPOINT}/v1/traces",
        headers={"x-confident-api-key": api_key},
    )
    provider.add_span_processor(RunIdSpanProcessor())
    provider.add_span_processor(BatchSpanProcessor(span_exporter=exporter))
    tracer = provider.get_tracer("deepeval_tracer")
    return provider, tracer

def init_global_test_run_tracer(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("CONFIDENT_API_KEY")
    if api_key is None:
        raise ValueError("CONFIDENT_API_KEY is not set")
    if not OTLP_ENDPOINT:
        raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT is not set")

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