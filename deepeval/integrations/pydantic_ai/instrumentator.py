from typing import Optional, List
from pydantic_ai.models.instrumented import InstrumentationSettings
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from deepeval.confident.api import get_confident_api_key

from deepeval.integrations.pydantic_ai.patcher import patch_agent_run

OTLP_ENDPOINT = "https://otel.confident-ai.com/v1/traces"

class ConfidentInstrumentationSettings(InstrumentationSettings):
    patch_agent_run()
    
    name: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None
    tags: Optional[List[str]] = None
    metric_collection: Optional[str] = None


    def __init__(self, api_key: Optional[str] = None):
        
        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError("CONFIDENT_API_KEY is not set")

        trace_provider = TracerProvider()
        trace_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=OTLP_ENDPOINT,
                    headers={"x-confident-api-key": api_key},
                )
            )
        )
        super().__init__(tracer_provider=trace_provider)