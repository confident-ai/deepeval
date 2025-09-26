import json
import os
from typing import Literal, Optional, List

try:
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    dependency_installed = True
except:
    dependency_installed = False


def is_dependency_installed():
    if not dependency_installed:
        raise ImportError(
            "Dependencies are not installed. Please install it with `pip install pydantic-ai opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`."
        )
    return True


from deepeval.confident.api import get_confident_api_key
from deepeval.prompt import Prompt

# OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
OTLP_ENDPOINT = "https://otel.confident-ai.com/v1/traces"


class SpanInterceptor(SpanProcessor):
    def __init__(self, settings_instance):
        # Keep a reference to the settings instance instead of copying values
        self.settings: ConfidentInstrumentationSettings = settings_instance

    def on_start(self, span, parent_context):

        # set trace attributes
        if self.settings.thread_id:
            span.set_attribute(
                "confident.trace.thread_id", self.settings.thread_id
            )
        if self.settings.user_id:
            span.set_attribute("confident.trace.user_id", self.settings.user_id)
        if self.settings.metadata:
            span.set_attribute(
                "confident.trace.metadata", json.dumps(self.settings.metadata)
            )
        if self.settings.tags:
            span.set_attribute("confident.trace.tags", self.settings.tags)
        if self.settings.metric_collection:
            span.set_attribute(
                "confident.trace.metric_collection",
                self.settings.metric_collection,
            )
        if self.settings.environment:
            span.set_attribute(
                "confident.trace.environment", self.settings.environment
            )
        if self.settings.name:
            span.set_attribute("confident.trace.name", self.settings.name)
        if self.settings.confident_prompt:
            span.set_attribute(
                "confident.span.prompt",
                json.dumps(
                    {
                        "alias": self.settings.confident_prompt.alias,
                        "version": self.settings.confident_prompt.version,
                    }
                ),
            )

        # set trace metric collection
        if self.settings.trace_metric_collection:
            span.set_attribute(
                "confident.trace.metric_collection",
                self.settings.trace_metric_collection,
            )

        # set agent name and metric collection
        if span.attributes.get("agent_name"):
            span.set_attribute("confident.span.type", "agent")
            span.set_attribute(
                "confident.span.name", span.attributes.get("agent_name")
            )
            if self.settings.agent_metric_collection:
                span.set_attribute(
                    "confident.span.metric_collection",
                    self.settings.agent_metric_collection,
                )

        # set llm metric collection
        if span.attributes.get("gen_ai.operation.name") in [
            "chat",
            "generate_content",
            "text_completion",
        ]:
            if self.settings.llm_metric_collection:
                span.set_attribute(
                    "confident.span.metric_collection",
                    self.settings.llm_metric_collection,
                )

        # set tool metric collection
        tool_name = span.attributes.get("gen_ai.tool.name")
        if tool_name:
            tool_metric_collection = (
                self.settings.tool_metric_collection_map.get(tool_name)
            )
            if tool_metric_collection:
                span.set_attribute(
                    "confident.span.metric_collection",
                    str(tool_metric_collection),
                )

    def on_end(self, span):
        pass


class ConfidentInstrumentationSettings(InstrumentationSettings):

    name: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None
    tags: Optional[List[str]] = None
    environment: Literal["production", "staging", "development", "testing"] = (
        None
    )
    metric_collection: Optional[str] = None
    confident_prompt: Optional[Prompt] = None
    llm_metric_collection: Optional[str] = None
    agent_metric_collection: Optional[str] = None
    tool_metric_collection_map: dict = {}
    trace_metric_collection: Optional[str] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        metric_collection: Optional[str] = None,
        confident_prompt: Optional[Prompt] = None,
        llm_metric_collection: Optional[str] = None,
        agent_metric_collection: Optional[str] = None,
        tool_metric_collection_map: dict = {},
        trace_metric_collection: Optional[str] = None,
    ):
        is_dependency_installed()

        _environment = os.getenv("CONFIDENT_TRACE_ENVIRONMENT", "development")
        if _environment and _environment in [
            "production",
            "staging",
            "development",
            "testing",
        ]:
            self.environment = _environment

        self.tool_metric_collection_map = tool_metric_collection_map
        self.name = name
        self.thread_id = thread_id
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.metric_collection = metric_collection
        self.confident_prompt = confident_prompt
        self.llm_metric_collection = llm_metric_collection
        self.agent_metric_collection = agent_metric_collection
        self.trace_metric_collection = trace_metric_collection

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
