import json
import logging
import os
from time import perf_counter
from typing import Optional, List, Dict

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.types import Trace, BaseSpan
from deepeval.tracing.otel.utils import to_hex_string
from deepeval.tracing.tracing import trace_manager


logger = logging.getLogger(__name__)


try:
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import ReadableSpan

    dependency_installed = True
except ImportError as e:
    if get_settings().DEEPEVAL_VERBOSE_MODE:
        if isinstance(e, ModuleNotFoundError):
            logger.warning(
                "Optional tracing dependency not installed: %s",
                e.name,
                stacklevel=2,
            )
        else:
            logger.warning(
                "Optional tracing import failed: %s",
                e,
                stacklevel=2,
            )
    dependency_installed = False


def is_dependency_installed():
    if not dependency_installed:
        raise ImportError(
            "Dependencies are not installed. Please install it with `pip install pydantic-ai opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`."
        )
    return True


from deepeval.tracing.types import AgentSpan
from deepeval.confident.api import get_confident_api_key
from deepeval.prompt import Prompt
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.types import Trace
from deepeval.tracing.otel.utils import to_hex_string
from deepeval.tracing.types import TraceSpanStatus, ToolCall
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.otel.exporter import ConfidentSpanExporter

# OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
OTLP_ENDPOINT = "https://otel.confident-ai.com/v1/traces"
init_clock_bridge()  # initialize clock bridge for perf_counter() to epoch_nanos conversion


class ConfidentInstrumentationSettings(InstrumentationSettings):

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
        tool_metric_collection_map: Optional[dict] = None,
        trace_metric_collection: Optional[str] = None,
        is_test_mode: Optional[bool] = False,
        agent_metrics: Optional[List[BaseMetric]] = None,
        llm_metrics: Optional[List[BaseMetric]] = None,
        tool_metrics: Optional[List[BaseMetric]] = None,
        trace_metrics: Optional[List[BaseMetric]] = None,
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

        self.tool_metric_collection_map = tool_metric_collection_map or {}
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
        self.is_test_mode = is_test_mode
        self.agent_metrics = agent_metrics
        self.llm_metrics = llm_metrics
        self.tool_metrics = tool_metrics
        self.trace_metrics = trace_metrics

        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError("CONFIDENT_API_KEY is not set")

        trace_provider = TracerProvider()

        # Pass the entire settings instance instead of individual values
        span_interceptor = SpanInterceptor(self)
        trace_provider.add_span_processor(span_interceptor)

        if is_test_mode:
            trace_provider.add_span_processor(BatchSpanProcessor(test_exporter))
        else:
            trace_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=OTLP_ENDPOINT,
                        headers={"x-confident-api-key": api_key},
                    )
                )
            )
        super().__init__(tracer_provider=trace_provider)


spans_by_trace_id: Dict[str, List[ReadableSpan]] = {}
agent_span_ids_by_trace_id: Dict[str, List[int]] = {}


class SpanInterceptor(SpanProcessor):
    def __init__(self, settings_instance: ConfidentInstrumentationSettings):
        # Keep a reference to the settings instance instead of copying values
        self.settings = settings_instance

    def on_start(self, span, parent_context):
        # track agent spans to find the root
        trace_id = span.context.trace_id
        print(trace_id)
        span_id = span.context.span_id
        is_agent_span = (
            span.attributes.get("agent_name") is not None
            if span.attributes
            else False
        )
        if is_agent_span:
            if trace_id not in agent_span_ids_by_trace_id:
                agent_span_ids_by_trace_id[trace_id] = []
            agent_span_ids_by_trace_id[trace_id].append(span_id)

        # set trace uuid
        _current_trace_context = current_trace_context.get()
        if _current_trace_context and isinstance(_current_trace_context, Trace):
            _otel_trace_id = span.get_span_context().trace_id
            _current_trace_context.uuid = to_hex_string(_otel_trace_id, 32)

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

    def on_end(
        self,
        span: ReadableSpan,
    ):
        trace_id = span.context.trace_id
        span_id = span.context.span_id
        agent_span_ids = agent_span_ids_by_trace_id.get(trace_id)
        if agent_span_ids is not None and span_id in agent_span_ids:
            agent_span_ids.remove(span_id)
        if not self.settings.is_test_mode:
            return

        # store spans by trace id
        if trace_id not in spans_by_trace_id:
            spans_by_trace_id[trace_id] = [span]
        else:
            spans_by_trace_id[trace_id].append(span)

        # detect root span
        if agent_span_ids is not None and len(agent_span_ids) == 0:
            print("Root span detected")
            # create a trace for evaluation if trace for uuid doesn't exist
            trace_uuid = to_hex_string(span.context.trace_id, 32)
            trace = trace_manager.get_trace_by_uuid(trace_uuid)
            if not trace:
                trace = trace_manager.start_new_trace(trace_uuid=trace_uuid)

            # add spans to trace in the correct order (parent before children)
            exporter = ConfidentSpanExporter()
            spans_in_trace: List[ReadableSpan] = spans_by_trace_id[trace_id]
            spans_in_trace.sort(key=lambda s: s.start_time)
            for otel_span in spans_in_trace:
                deepeval_span_wrapper = (
                    exporter._convert_readable_span_to_base_span(otel_span)
                )
                deepeval_span = deepeval_span_wrapper.base_span
                self._set_metrics(
                    deepeval_span,
                    otel_span.attributes.get("confident.span.type"),
                )
                if span_id == otel_span.context.span_id:
                    deepeval_span.parent_uuid = None
                # set parent uuid for root span
                trace_manager.add_span(deepeval_span)
                trace_manager.add_span_to_trace(deepeval_span)

            # Finalize trace for evaluation
            trace.status = TraceSpanStatus.SUCCESS
            trace.end_time = perf_counter()
            trace_manager.traces_to_evaluate.append(trace)
            print(len(trace_manager.traces_to_evaluate))
            print(trace_manager.traces_to_evaluate)
            del spans_by_trace_id[trace_id]
            del agent_span_ids_by_trace_id[trace_id]

    def _set_metrics(
        self,
        base_span: BaseSpan,
        span_type: str,
    ):
        if span_type == "agent":
            if self.settings.agent_metrics:
                base_span.metrics = self.settings.agent_metrics
        elif span_type == "llm":
            if self.settings.llm_metrics:
                base_span.metrics = self.settings.llm_metrics
        elif span_type == "tool":
            if self.settings.tool_metrics:
                base_span.metrics = self.settings.tool_metrics
