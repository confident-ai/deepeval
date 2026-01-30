from __future__ import annotations

import json
import logging
import os
from time import perf_counter
from typing import Any, List, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.otel.utils import (
    normalize_pydantic_ai_messages,
    to_hex_string,
)
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    Trace,
    TraceSpanStatus,
    ToolCall,
)

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    # Optional dependencies
    from opentelemetry.sdk.trace import (
        ReadableSpan as _ReadableSpan,
        SpanProcessor as _SpanProcessor,
        TracerProvider,
    )
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.trace import set_tracer_provider
    from pydantic_ai.models.instrumented import (
        InstrumentationSettings as _BaseInstrumentationSettings,
    )

    dependency_installed = True
except ImportError as e:
    dependency_installed = False

    # Preserve previous behavior: only log when verbose mode is enabled.
    if settings.DEEPEVAL_VERBOSE_MODE:
        if isinstance(e, ModuleNotFoundError):
            logger.warning(
                "Optional tracing dependency not installed: %s",
                getattr(e, "name", repr(e)),
                stacklevel=2,
            )
        else:
            logger.warning(
                "Optional tracing import failed: %s",
                e,
                stacklevel=2,
            )

    # Dummy fallbacks so imports and class definitions don't crash when
    # optional deps are missing. Actual use is still guarded by
    # is_dependency_installed().
    class _BaseInstrumentationSettings:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _SpanProcessor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_start(self, span: Any, parent_context: Any) -> None:
            pass

        def on_end(self, span: Any) -> None:
            pass

    class _ReadableSpan:
        pass


def is_dependency_installed() -> bool:
    if not dependency_installed:
        raise ImportError(
            "Dependencies are not installed. Please install it with "
            "`pip install pydantic-ai opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http`."
        )
    return True


if TYPE_CHECKING:
    # For type checkers, use real types
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
    from pydantic_ai.models.instrumented import InstrumentationSettings
else:
    # At runtime we always have something to subclass / annotate with
    InstrumentationSettings = _BaseInstrumentationSettings
    SpanProcessor = _SpanProcessor
    ReadableSpan = _ReadableSpan

# OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
# OTLP_ENDPOINT = "https://otel.confident-ai.com/v1/traces"
OTLP_ENDPOINT = str(settings.CONFIDENT_OTEL_URL) + "v1/traces"
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
    ):
        is_dependency_installed()

        if trace_manager.environment is not None:
            _environment = trace_manager.environment
        elif settings.CONFIDENT_TRACE_ENVIRONMENT is not None:
            _environment = settings.CONFIDENT_TRACE_ENVIRONMENT
        else:
            _environment = "development"
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

        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError("CONFIDENT_API_KEY is not set")

        trace_provider = TracerProvider()

        # Pass the entire settings instance instead of individual values
        span_interceptor = SpanInterceptor(self)
        trace_provider.add_span_processor(span_interceptor)

        if is_test_mode:
            trace_provider.add_span_processor(
                SimpleSpanProcessor(ConfidentSpanExporter())
            )
        else:
            trace_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=OTLP_ENDPOINT,
                        headers={"x-confident-api-key": api_key},
                    )
                )
            )
        try:
            set_tracer_provider(trace_provider)
        except Exception as e:
            # Handle case where provider is already set (optional warning)
            logger.warning(f"Could not set global tracer provider: {e}")

        super().__init__(tracer_provider=trace_provider)


class SpanInterceptor(SpanProcessor):
    def __init__(self, settings_instance: ConfidentInstrumentationSettings):
        # Keep a reference to the settings instance instead of copying values
        self.settings = settings_instance

    def on_start(self, span, parent_context):
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
        agent_name = (
            span.attributes.get("gen_ai.agent.name")
            or span.attributes.get("pydantic_ai.agent.name")
            or span.attributes.get("agent_name")
        )

        if agent_name:
            self._add_agent_span(span, agent_name)

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

        already_processed = (
            span.attributes.get("confident.span.type") == "agent"
        )
        if not already_processed:
            agent_name = (
                span.attributes.get("gen_ai.agent.name")
                or span.attributes.get("pydantic_ai.agent.name")
                or span.attributes.get("agent_name")
            )
            if agent_name:
                self._add_agent_span(span, agent_name)

        if self.settings.is_test_mode:
            if span.attributes.get("confident.span.type") == "agent":

                def create_agent_span_for_evaluation(
                    span: ReadableSpan,
                ) -> AgentSpan:

                    agent_span = (
                        ConfidentSpanExporter.prepare_boilerplate_base_span(
                            span
                        )
                    )

                    # tools called
                    normalized_messages = normalize_pydantic_ai_messages(span)
                    tools_called = []

                    for message in normalized_messages:
                        for part in message.get("parts", []):
                            if part.get("type") == "tool_call":
                                name = part.get("name")
                                try:
                                    input_parameters = json.loads(
                                        part.get("arguments")
                                    )
                                except Exception:
                                    input_parameters = {}

                                tools_called.append(
                                    ToolCall(
                                        name=name,
                                        input_parameters=input_parameters,
                                    )
                                )

                    # agent_span.tools_called = tools_called
                    return agent_span

                agent_span = create_agent_span_for_evaluation(span)
                agent_span.metrics = self.settings.agent_metrics

                # create a trace for evaluation
                trace = trace_manager.get_trace_by_uuid(agent_span.trace_uuid)
                if not trace:
                    trace = trace_manager.start_new_trace(
                        trace_uuid=agent_span.trace_uuid
                    )

                trace.root_spans.append(agent_span)
                trace.status = TraceSpanStatus.SUCCESS
                trace.end_time = perf_counter()
                trace_manager.traces_to_evaluate.append(trace)

    def _add_agent_span(self, span, name):
        span.set_attribute("confident.span.type", "agent")
        span.set_attribute("confident.span.name", name)
        if self.settings.agent_metric_collection:
            span.set_attribute(
                "confident.span.metric_collection",
                self.settings.agent_metric_collection,
            )
