from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, List, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from deepeval.tracing.otel.utils import to_hex_string
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

    dependency_installed = True
except ImportError as e:
    dependency_installed = False

    if settings.DEEPEVAL_VERBOSE_MODE:
        logger.warning(
            "Optional tracing dependency not installed: %s",
            getattr(e, "name", repr(e)),
            stacklevel=2,
        )

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
            "Dependencies are not installed. Please install them with "
            "`pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`."
        )
    return True


if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
else:
    SpanProcessor = _SpanProcessor
    ReadableSpan = _ReadableSpan

OTLP_ENDPOINT = str(settings.CONFIDENT_OTEL_URL) + "v1/traces"
init_clock_bridge()

_AGENT_OP_NAMES = {"invoke_agent", "create_agent"}

# gen_ai.operation.name values that indicate an LLM-call span
_LLM_OP_NAMES = {
    "chat",
    "generate_content",
    "invoke_model",
    "text_completion",
    "embeddings",
}

# gen_ai.operation.name values that indicate a tool span
_TOOL_OP_NAMES = {"execute_tool"}

# traceloop.span.kind values → confident span type
_TRACELOOP_KIND_MAP = {
    "workflow": "agent",
    "agent": "agent",
    "task": "tool",
    "tool": "tool",
    "retriever": "retriever",
    "llm": "llm",
}


def _get_attr(span, *keys: str) -> Optional[str]:
    attrs = span.attributes or {}
    for k in keys:
        v = attrs.get(k)
        if v:
            return str(v)
    return None


def _classify_span(span) -> Optional[str]:
    attrs = span.attributes or {}
    span_name_lower = (span.name or "").lower()

    # --- 1. Explicit gen_ai.operation.name (Strands + generic OTel) ---
    op_name = attrs.get("gen_ai.operation.name", "")
    if op_name in _AGENT_OP_NAMES:
        return "agent"
    if op_name in _LLM_OP_NAMES:
        return "llm"
    if op_name in _TOOL_OP_NAMES:
        return "tool"

    # --- 2. OpenLLMetry / traceloop conventions (LangChain, LangGraph, CrewAI) ---
    traceloop_kind = attrs.get("traceloop.span.kind", "")
    if traceloop_kind in _TRACELOOP_KIND_MAP:
        return _TRACELOOP_KIND_MAP[traceloop_kind]

    # --- 3. Presence of canonical tool/agent attributes ---
    if attrs.get("gen_ai.tool.name") or attrs.get("gen_ai.tool.call.id"):
        return "tool"
    if attrs.get("gen_ai.agent.name") or attrs.get("gen_ai.agent.id"):
        return "agent"

    # --- 4. Heuristic span-name matching (last resort) ---
    if any(kw in span_name_lower for kw in ("invoke_agent", "agent")):
        return "agent"
    if any(kw in span_name_lower for kw in ("execute_tool", ".tool")):
        return "tool"
    if any(kw in span_name_lower for kw in ("retriev", "memory", "datastore")):
        return "retriever"
    if any(
        kw in span_name_lower
        for kw in ("llm", "chat", "invoke_model", "generate")
    ):
        return "llm"

    return None


def _get_agent_name(span) -> Optional[str]:
    """Extract the most descriptive agent name available."""
    return (
        _get_attr(
            span,
            "gen_ai.agent.name",
            "traceloop.entity.name",
            "traceloop.workflow.name",
        )
        or span.name
        or None
    )


def _get_tool_name(span) -> Optional[str]:
    """Extract the tool name from a tool span."""
    return (
        _get_attr(
            span,
            "gen_ai.tool.name",
            "traceloop.entity.name",
        )
        or span.name
        or None
    )


# ---------------------------------------------------------------------------
# Content / I/O extraction helpers
# ---------------------------------------------------------------------------


def _parse_genai_content(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return str(raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return first.get("text") or first.get("content") or str(first)
            return str(first)
        if isinstance(data, dict):
            return data.get("text") or data.get("content") or str(data)
        return str(data)
    except (json.JSONDecodeError, TypeError):
        return raw


def _extract_messages(span) -> tuple[Optional[str], Optional[str]]:
    input_text: Optional[str] = None
    output_text: Optional[str] = None

    # --- 1. Extract from Events (Strands / strict OTel GenAI) ---
    for event in getattr(span, "events", []):
        event_name = event.name or ""
        event_attrs = event.attributes or {}

        if event_name == "gen_ai.user.message":
            input_text = _parse_genai_content(event_attrs.get("content"))
        elif event_name in ("gen_ai.choice", "gen_ai.assistant.message"):
            output_text = _parse_genai_content(
                event_attrs.get("message") or event_attrs.get("content")
            )
        elif event_name == "gen_ai.system.message":
            if not input_text:
                input_text = _parse_genai_content(event_attrs.get("content"))
        elif event_name in (
            "gen_ai.client.inference.operation.details",
            "agent.invocation",
            "tool.invocation",
        ):
            body_raw = event_attrs.get("body") or event_attrs.get("event.body")
            if body_raw:
                try:
                    body = (
                        json.loads(body_raw)
                        if isinstance(body_raw, str)
                        else body_raw
                    )
                    if not input_text and "input" in body:
                        msgs = body["input"].get("messages", [])
                        if msgs:
                            input_text = _parse_genai_content(
                                msgs[-1].get("content")
                                if isinstance(msgs[-1], dict)
                                else msgs[-1]
                            )
                    if not output_text and "output" in body:
                        msgs = body["output"].get("messages", [])
                        if msgs:
                            output_text = _parse_genai_content(
                                msgs[-1].get("content")
                                if isinstance(msgs[-1], dict)
                                else msgs[-1]
                            )
                except Exception:
                    pass

    # --- 2. Fall back to attributes (LangChain, CrewAI, Traceloop) ---
    if not input_text:
        raw = _get_attr(
            span,
            "gen_ai.user.message",
            "gen_ai.input.messages",
            "gen_ai.prompt",
            "traceloop.entity.input",
            "crewai.task.description",
        )
        if raw:
            input_text = _parse_genai_content(raw)

    if not output_text:
        raw = _get_attr(
            span,
            "gen_ai.choice",
            "gen_ai.output.messages",
            "gen_ai.completion",
            "traceloop.entity.output",
        )
        if raw:
            output_text = _parse_genai_content(raw)

    return input_text, output_text


def _extract_tool_calls(span) -> List[ToolCall]:
    tools: List[ToolCall] = []

    # --- 1. Extract from events (Strands / strict OTel) ---
    for event in getattr(span, "events", []):
        event_attrs = event.attributes or {}
        event_name = event.name or ""

        if event_name in ("gen_ai.tool.call", "tool_call", "execute_tool"):
            try:
                name = (
                    event_attrs.get("gen_ai.tool.name")
                    or event_attrs.get("name")
                    or "unknown_tool"
                )
                args_raw = (
                    event_attrs.get("gen_ai.tool.call.arguments")
                    or event_attrs.get("gen_ai.tool.arguments")
                    or event_attrs.get("input")
                    or "{}"
                )
                input_params = (
                    json.loads(args_raw)
                    if isinstance(args_raw, str)
                    else args_raw
                )
                tools.append(
                    ToolCall(name=str(name), input_parameters=input_params)
                )
            except Exception as exc:
                logger.debug("Failed to parse tool call event: %s", exc)

    # --- 2. Extract from attributes (LangChain / CrewAI / Traceloop) ---
    attrs = span.attributes or {}

    tool_calls_raw = (
        attrs.get("gen_ai.tool.calls")
        or attrs.get("traceloop.tool_calls")
        or attrs.get("llm.tool_calls")
    )

    if tool_calls_raw:
        try:
            calls = (
                json.loads(tool_calls_raw)
                if isinstance(tool_calls_raw, str)
                else tool_calls_raw
            )
            if isinstance(calls, list):
                for call in calls:
                    # Traceloop/OpenLLMetry often nests these under a "function" key
                    name = (
                        call.get("name")
                        or call.get("function", {}).get("name")
                        or "unknown_tool"
                    )
                    args = (
                        call.get("arguments")
                        or call.get("function", {}).get("arguments")
                        or "{}"
                    )

                    input_params = (
                        json.loads(args) if isinstance(args, str) else args
                    )
                    tools.append(
                        ToolCall(name=str(name), input_parameters=input_params)
                    )
        except Exception as exc:
            logger.debug("Failed to parse tool call attributes: %s", exc)

    return tools


def _extract_tool_call_from_tool_span(span) -> Optional[ToolCall]:
    tool_name = _get_tool_name(span)
    if not tool_name:
        return None

    attrs = span.attributes or {}
    args_raw = (
        attrs.get("gen_ai.tool.call.arguments")
        or attrs.get("traceloop.entity.input")
        or "{}"
    )
    try:
        input_params = (
            json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        )
    except Exception:
        input_params = {}

    return ToolCall(name=tool_name, input_parameters=input_params)


class AgentCoreInstrumentationSettings:

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
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        is_test_mode: Optional[bool] = False,
        agent_metrics: Optional[List[BaseMetric]] = None,
        environment: Optional[str] = None,
    ):
        is_dependency_installed()

        if trace_manager.environment is not None:
            _env = trace_manager.environment
        elif environment is not None:
            _env = environment
        elif settings.CONFIDENT_TRACE_ENVIRONMENT is not None:
            _env = settings.CONFIDENT_TRACE_ENVIRONMENT
        else:
            _env = "development"

        if _env not in ("production", "staging", "development", "testing"):
            _env = "development"
        self.environment = _env

        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError(
                    "CONFIDENT_API_KEY is not set. Pass pass api_key or set CONFIDENT_API_KEY in your env"
                )
        self.api_key = api_key

        self.name = name
        self.thread_id = thread_id
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.metric_collection = metric_collection
        self.confident_prompt = confident_prompt
        self.llm_metric_collection = llm_metric_collection
        self.agent_metric_collection = agent_metric_collection
        self.tool_metric_collection_map = tool_metric_collection_map or {}
        self.trace_metric_collection = trace_metric_collection
        self.test_case_id = test_case_id
        self.turn_id = turn_id
        self.is_test_mode = is_test_mode
        self.agent_metrics = agent_metrics


class AgentCoreSpanInterceptor(SpanProcessor):

    def __init__(self, settings_instance: AgentCoreInstrumentationSettings):
        self.settings = settings_instance

    def on_start(self, span, parent_context):
        settings = self.settings

        _ctx = current_trace_context.get()
        if _ctx and isinstance(_ctx, Trace):
            _ctx.uuid = to_hex_string(span.get_span_context().trace_id, 32)

        _safe_set = lambda k, v: span.set_attribute(k, v) if v else None

        _safe_set("confident.trace.name", settings.name)
        _safe_set("confident.trace.thread_id", settings.thread_id)
        _safe_set("confident.trace.user_id", settings.user_id)
        _safe_set("confident.trace.environment", settings.environment)
        _safe_set("confident.trace.test_case_id", settings.test_case_id)
        _safe_set("confident.trace.turn_id", settings.turn_id)

        if settings.metadata:
            span.set_attribute(
                "confident.trace.metadata", json.dumps(settings.metadata)
            )
        if settings.tags:
            span.set_attribute("confident.trace.tags", settings.tags)

        metric_collection = (
            settings.trace_metric_collection or settings.metric_collection
        )
        _safe_set("confident.trace.metric_collection", metric_collection)

        if settings.confident_prompt:
            prompt = settings.confident_prompt
            span.set_attribute("confident.span.prompt_alias", prompt.alias)
            span.set_attribute("confident.span.prompt_commit_hash", prompt.hash)
            if getattr(prompt, "label", None):
                span.set_attribute("confident.span.prompt_label", prompt.label)
            if getattr(prompt, "version", None):
                span.set_attribute(
                    "confident.span.prompt_version", prompt.version
                )

        span_type = _classify_span(span)
        if span_type is None:
            return

        span.set_attribute("confident.span.type", span_type)

        if span_type == "agent":
            agent_name = _get_agent_name(span)
            if agent_name:
                span.set_attribute("confident.span.name", agent_name)
            if settings.agent_metric_collection:
                span.set_attribute(
                    "confident.span.metric_collection",
                    settings.agent_metric_collection,
                )

        elif span_type == "llm":
            if settings.llm_metric_collection:
                span.set_attribute(
                    "confident.span.metric_collection",
                    settings.llm_metric_collection,
                )

        elif span_type == "tool":
            tool_name = _get_tool_name(span)
            if tool_name:
                tool_mc = settings.tool_metric_collection_map.get(tool_name)
                if tool_mc:
                    span.set_attribute(
                        "confident.span.metric_collection", tool_mc
                    )

        if not settings.thread_id:
            session_id = (span.attributes or {}).get("session.id")
            if session_id:
                span.set_attribute("confident.trace.thread_id", session_id)

    def on_end(self, span: ReadableSpan):
        attrs = dict(span.attributes or {})

        if "confident.span.type" not in attrs:
            span_type = _classify_span(span)
            if span_type:
                span._attributes["confident.span.type"] = span_type
                attrs["confident.span.type"] = span_type

        span_type = attrs.get("confident.span.type")

        input_text, output_text = _extract_messages(span)

        if input_text:
            span._attributes["confident.span.input"] = input_text
            if span_type == "agent":
                span._attributes["confident.trace.input"] = input_text

        if output_text:
            span._attributes["confident.span.output"] = output_text
            if span_type == "agent":
                span._attributes["confident.trace.output"] = output_text

        input_tokens = attrs.get("gen_ai.usage.input_tokens") or attrs.get(
            "gen_ai.usage.prompt_tokens"
        )
        output_tokens = attrs.get("gen_ai.usage.output_tokens") or attrs.get(
            "gen_ai.usage.completion_tokens"
        )
        if input_tokens is not None:
            span._attributes["confident.llm.input_token_count"] = int(
                input_tokens
            )
        if output_tokens is not None:
            span._attributes["confident.llm.output_token_count"] = int(
                output_tokens
            )

        model = _get_attr(
            span,
            "gen_ai.response.model",
            "gen_ai.request.model",
        )
        if model:
            span._attributes["confident.llm.model"] = model

        tools_called: List[ToolCall] = []

        if span_type == "agent":
            tools_called = _extract_tool_calls(span)

            tool_defs_raw = attrs.get("gen_ai.tool.definitions") or attrs.get(
                "gen_ai.agent.tools"
            )
            if tool_defs_raw:
                span._attributes["confident.agent.tool_definitions"] = str(
                    tool_defs_raw
                )

        elif span_type == "tool":
            tc = _extract_tool_call_from_tool_span(span)
            if tc:
                tools_called = [tc]

                if tc.input_parameters and not input_text:
                    span._attributes["confident.span.input"] = json.dumps(
                        tc.input_parameters
                    )

            if not output_text:
                raw_output = _get_attr(
                    span, "traceloop.entity.output", "gen_ai.tool.output"
                )
                if raw_output:
                    span._attributes["confident.span.output"] = raw_output

        if tools_called:
            span._attributes["confident.span.tools_called"] = [
                t.model_dump_json() for t in tools_called
            ]

        if (
            span_type == "agent"
            and "confident.span.name" not in span._attributes
        ):
            agent_name = _get_agent_name(span)
            if agent_name:
                span._attributes["confident.span.name"] = agent_name

        if self.settings.is_test_mode and span_type == "agent":
            self._handle_test_mode(span, tools_called)

    def _handle_test_mode(
        self, span: ReadableSpan, tools_called: List[ToolCall] = None
    ) -> None:
        """Build an AgentSpan for evaluation and register it with trace_manager."""
        try:
            agent_span: Optional[AgentSpan] = (
                ConfidentSpanExporter.prepare_boilerplate_base_span(span)
            )
        except Exception as exc:
            logger.debug("prepare_boilerplate_base_span failed: %s", exc)
            return

        if not agent_span:
            return

        attrs = dict(span.attributes or {})
        input = attrs.get("confident.span.input") or span._attributes.get(
            "confident.span.input"
        )
        output = attrs.get("confident.span.output") or span._attributes.get(
            "confident.span.output"
        )

        if input and not getattr(agent_span, "input", None):
            agent_span.input = input
        if output and not getattr(agent_span, "output", None):
            agent_span.output = output

        agent_span.tools_called = tools_called
        agent_span.metrics = self.settings.agent_metrics

        active_trace = current_trace_context.get()

        if active_trace and isinstance(active_trace, Trace):
            trace = active_trace
            if not trace.uuid:
                trace.uuid = agent_span.trace_uuid
        else:
            trace = trace_manager.get_trace_by_uuid(agent_span.trace_uuid)
            if not trace:
                trace = trace_manager.start_new_trace(
                    trace_uuid=agent_span.trace_uuid
                )

        if agent_span.input and not getattr(trace, "input", None):
            trace.input = agent_span.input
        if agent_span.output and not getattr(trace, "output", None):
            trace.output = agent_span.output

        trace.root_spans.append(agent_span)
        trace.status = TraceSpanStatus.SUCCESS
        trace.end_time = perf_counter()

        if trace not in trace_manager.traces_to_evaluate:
            trace_manager.traces_to_evaluate.append(trace)
