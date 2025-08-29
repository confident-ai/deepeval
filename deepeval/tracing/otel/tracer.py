import os
from typing import Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# OTLP_ENDPOINT = "http://127.0.0.1:4318"
OTLP_ENDPOINT = "https://otel.confident-ai.com"


class TracerManager:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key if api_key is not None else os.getenv("CONFIDENT_API_KEY")

        if not api_key:
            raise ValueError(
                "CONFIDENT_API_KEY is not set. Either pass it as an argument or set it as an environment variable."
            )

        self.tracer_provider = TracerProvider()
        exporter = OTLPSpanExporter(
            endpoint=f"{OTLP_ENDPOINT}/v1/traces",
            headers={"x-confident-api-key": api_key},
        )
        span_processor = BatchSpanProcessor(span_exporter=exporter)
        self.tracer_provider.add_span_processor(span_processor)

        self.tracer = self.tracer_provider.get_tracer("deepeval_tracer")
    
    def get_tracer(self) -> Tracer:
        return self.tracer
    
    def get_tracer_provider(self) -> TracerProvider:
        return self.tracer_provider
