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
    )

    dependency_installed = True
except ImportError as e:
    dependency_installed = False

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

init_clock_bridge()

# ---------------------------------------------------------------------------
# OpenInference Extraction Helpers
# ---------------------------------------------------------------------------


def _get_span_kind(span) -> Optional[str]:
    attrs = span.attributes if hasattr(span, "attributes") else span._attributes
    kind = str(attrs.get("openinference.span.kind", "")).upper()

    if not kind:
        return None

    if kind == "AGENT" or kind == "CHAIN":
        return "agent"
    elif kind == "LLM":
        return "llm"
    elif kind == "TOOL":
        return "tool"
    elif kind == "RETRIEVER":
        return "retriever"

    return "custom"


def _extract_messages(span) -> tuple[Optional[str], Optional[str]]:
    attrs = (
        getattr(span, "attributes", None)
        or getattr(span, "_attributes", None)
        or {}
    )

    input_text = None
    output_text = None

    # 1. EXTRACT INPUTS following the OpenInference semantic conventions for LLMs
    idx = 0
    last_content = None
    while True:
        role_key = f"llm.input_messages.{idx}.message.role"
        content_key = f"llm.input_messages.{idx}.message.content"
        if role_key in attrs or content_key in attrs:
            content = attrs.get(content_key)
            if content is not None:
                last_content = content
            idx += 1
        else:
            break

    if last_content is not None:
        input_text = last_content

    elif "llm.input_messages" in attrs:
        try:
            raw_msgs = attrs["llm.input_messages"]
            data = (
                json.loads(raw_msgs) if isinstance(raw_msgs, str) else raw_msgs
            )
            if isinstance(data, list) and len(data) > 0:
                last_msg = data[-1]
                input_text = (
                    last_msg.get("content")
                    or last_msg.get("message", {}).get("content")
                    or str(last_msg)
                )
        except Exception:
            input_text = str(attrs["llm.input_messages"])

    # Pure generic fallback for Agents/Tools
    if not input_text:
        input_text = attrs.get("input.value")

    # 2. EXTRACT OUTPUTS
    idx = 0
    last_content = None
    while True:
        role_key = f"llm.output_messages.{idx}.message.role"
        content_key = f"llm.output_messages.{idx}.message.content"
        if role_key in attrs or content_key in attrs:
            content = attrs.get(content_key)
            if content is not None:
                last_content = content
            idx += 1
        else:
            break

    if last_content is not None:
        output_text = last_content

    elif "llm.output_messages" in attrs:
        try:
            raw_msgs = attrs["llm.output_messages"]
            data = (
                json.loads(raw_msgs) if isinstance(raw_msgs, str) else raw_msgs
            )
            if isinstance(data, list) and len(data) > 0:
                last_msg = data[-1]
                output_text = (
                    last_msg.get("content")
                    or last_msg.get("message", {}).get("content")
                    or str(last_msg)
                )
        except Exception:
            output_text = str(attrs["llm.output_messages"])

    # Pure generic fallback for Agents/Tools
    if not output_text:
        output_text = attrs.get("output.value")

    return (
        str(input_text) if input_text is not None else None,
        str(output_text) if output_text is not None else None,
    )


def _extract_tool_calls(span) -> List[ToolCall]:
    attrs = (
        span._attributes if hasattr(span, "_attributes") else span.attributes
    )
    tools: List[ToolCall] = []

    # Scenario A: The span itself is a Tool
    if "tool.name" in attrs:
        tool_name = attrs.get("tool.name")
        tool_args = attrs.get("tool.parameters") or "{}"
        try:
            params = (
                json.loads(tool_args)
                if isinstance(tool_args, str)
                else tool_args
            )
        except Exception:
            params = {}
        tools.append(ToolCall(name=str(tool_name), input_parameters=params))
        return tools

    # Scenario B: The span is an LLM span with tool calls in output_messages
    msg_idx = 0
    while True:
        # Check if an output message exists at this index
        if (
            f"llm.output_messages.{msg_idx}.message.role" not in attrs
            and f"llm.output_messages.{msg_idx}.message.content" not in attrs
        ):
            break

        tc_idx = 0
        while True:
            # Flattened convention for nested tool calls
            base_key = f"llm.output_messages.{msg_idx}.message.tool_calls.{tc_idx}.tool_call.function"
            name_key = f"{base_key}.name"

            if name_key in attrs:
                t_name = attrs[name_key]
                t_args = attrs.get(f"{base_key}.arguments", "{}")
                try:
                    t_params = (
                        json.loads(t_args)
                        if isinstance(t_args, str)
                        else t_args
                    )
                except Exception:
                    t_params = {}
                tools.append(
                    ToolCall(name=str(t_name), input_parameters=t_params)
                )
                tc_idx += 1
            else:
                break

        msg_idx += 1

    # Fallback to unflattened output_messages if tools wasn't populated
    if not tools and "llm.output_messages" in attrs:
        try:
            raw_msgs = attrs["llm.output_messages"]
            data = (
                json.loads(raw_msgs) if isinstance(raw_msgs, str) else raw_msgs
            )
            if isinstance(data, list):
                for msg in data:
                    for tc in msg.get("tool_calls", []):
                        func = tc.get("function", {})
                        t_name = func.get("name")
                        t_args = func.get("arguments", "{}")
                        if t_name:
                            try:
                                t_params = (
                                    json.loads(t_args)
                                    if isinstance(t_args, str)
                                    else t_args
                                )
                            except Exception:
                                t_params = {}
                            tools.append(
                                ToolCall(
                                    name=str(t_name), input_parameters=t_params
                                )
                            )
        except Exception:
            pass

    return tools


class OpenInferenceInstrumentationSettings:
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
                    "CONFIDENT_API_KEY is not set. Pass api_key or set CONFIDENT_API_KEY in your env"
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


class OpenInferenceSpanInterceptor(SpanProcessor):
    def __init__(self, settings_instance: OpenInferenceInstrumentationSettings):
        self.settings = settings_instance

    def on_start(self, span, parent_context):
        settings = self.settings

        _ctx = current_trace_context.get()
        if _ctx and isinstance(_ctx, Trace):
            _ctx.uuid = to_hex_string(span.get_span_context().trace_id, 32)

        _safe_set = lambda k, v: span.set_attribute(k, v) if v else None

        _safe_set("confident.trace.name", settings.name)
        _safe_set("confident.trace.environment", settings.environment)
        _safe_set("confident.trace.thread_id", settings.thread_id)
        _safe_set("confident.trace.user_id", settings.user_id)
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

        span_type = _get_span_kind(span)
        if span_type is None:
            return

        span.set_attribute("confident.span.type", span_type)

        if span_type == "agent":
            agent_name = span.attributes.get("agent.name") or span.name
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
            tool_name = span.attributes.get("tool.name") or span.name
            if tool_name:
                span.set_attribute("confident.span.name", tool_name)
                tool_mc = settings.tool_metric_collection_map.get(tool_name)
                if tool_mc:
                    span.set_attribute(
                        "confident.span.metric_collection", tool_mc
                    )

    def on_end(self, span: ReadableSpan):
        if getattr(span, "_attributes", None) is None:
            span._attributes = {}

        attrs = span._attributes

        if "confident.span.type" not in attrs:
            span_type = _get_span_kind(span)
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

        # Standard Token usage keys
        input_tokens = attrs.get("llm.token_count.prompt")
        output_tokens = attrs.get("llm.token_count.completion")

        if input_tokens is not None:
            span._attributes["confident.llm.input_token_count"] = int(
                input_tokens
            )
        if output_tokens is not None:
            span._attributes["confident.llm.output_token_count"] = int(
                output_tokens
            )

        model = attrs.get("llm.model_name")  # Capture the exact model string
        if model:
            span._attributes["confident.llm.model"] = str(model)

        tools_called: List[ToolCall] = []

        if span_type in ("agent", "tool", "llm"):
            tools_called = _extract_tool_calls(span)

        if tools_called:
            span._attributes["confident.span.tools_called"] = [
                (
                    t.model_dump_json()
                    if hasattr(t, "model_dump_json")
                    else json.dumps(t)
                )
                for t in tools_called
            ]

        if self.settings.is_test_mode and span_type == "agent":
            self._handle_test_mode(span, tools_called)

    def _handle_test_mode(
        self, span: ReadableSpan, tools_called: List[ToolCall] = None
    ) -> None:
        try:
            agent_span: Optional[AgentSpan] = (
                ConfidentSpanExporter.prepare_boilerplate_base_span(span)
            )
        except Exception as exc:
            logger.debug("prepare_boilerplate_base_span failed: %s", exc)
            return

        if not agent_span:
            return

        attrs = span._attributes
        input_val = attrs.get("confident.span.input") or span._attributes.get(
            "confident.span.input"
        )
        output_val = attrs.get("confident.span.output") or span._attributes.get(
            "confident.span.output"
        )

        if input_val and not getattr(agent_span, "input", None):
            agent_span.input = input_val
        if output_val and not getattr(agent_span, "output", None):
            agent_span.output = output_val

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
