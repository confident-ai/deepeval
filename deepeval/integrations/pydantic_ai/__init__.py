from typing import Optional
import deepeval

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

from deepeval.tracing.otel.exporter import ConfidentSpanExporter

def setup_instrumentation(api_key: Optional[str] = None):

    if api_key:
        deepeval.login_with_confident_api_key(api_key)
    
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
    else:
        tracer_provider = trace.get_tracer_provider()

    exporter = ConfidentSpanExporter()
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)
    set_tracer_provider(tracer_provider)