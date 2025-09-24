import json
from typing import Literal, Optional, List
from pydantic_ai.models.instrumented import InstrumentationSettings
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, Tracer
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from deepeval.confident.api import get_confident_api_key

OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"

class SpanInterceptor(SpanProcessor):
    def __init__(self, settings_instance):
        # Keep a reference to the settings instance instead of copying values
        self.settings: ConfidentInstrumentationSettings = settings_instance

    def on_start(self, span, parent_context):
        if self.settings.thread_id:
            span.set_attribute("confident.trace.thread_id", self.settings.thread_id)
        if self.settings.user_id:
            span.set_attribute("confident.trace.user_id", self.settings.user_id)
        if self.settings.metadata:
            span.set_attribute("confident.trace.metadata", json.dumps(self.settings.metadata))
        if self.settings.tags:
            span.set_attribute("confident.trace.tags", self.settings.tags)
        if self.settings.metric_collection:
            span.set_attribute("confident.trace.metric_collection", self.settings.metric_collection)
        if self.settings.environment:
            span.set_attribute("confident.trace.environment", self.settings.environment)
        if self.settings.name:
            span.set_attribute("confident.trace.name", self.settings.name)
        if span.attributes.get("agent_name"):
            span.set_attribute("confident.span.type", "agent")
    
    def on_end(self, span):
        pass
    

class ConfidentInstrumentationSettings(InstrumentationSettings):
    
    name: Optional[str] = None 
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None
    tags: Optional[List[str]] = None
    environment: Literal["production", "staging", "development", "testing"] = None
    metric_collection: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None):
        
        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError("CONFIDENT_API_KEY is not set")

        trace_provider = TracerProvider()
        
        # Pass the entire settings instance instead of individual values
        span_interceptor = SpanInterceptor(self)
        trace_provider.add_span_processor(span_interceptor)
        
        trace_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=OTLP_ENDPOINT,
                    headers={"x-confident-api-key": api_key},
                )
            )
        )
        super().__init__(tracer_provider=trace_provider)
    
    def set_trace_attributes(
        self,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        metric_collection: Optional[str] = None,
        environment: Literal["production", "staging", "development", "testing"] = None,
        name: Optional[str] = None,
    ):
        if thread_id:   
            self.thread_id = thread_id
        if user_id:
            self.user_id = user_id
        if metadata:
            self.metadata = metadata
        if tags:
            self.tags = tags
        if metric_collection:
            self.metric_collection = metric_collection
        if environment:
            self.environment = environment
        if name:
            self.name = name