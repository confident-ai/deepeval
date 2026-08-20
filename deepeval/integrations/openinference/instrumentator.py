"""OpenInference × deepeval OTel SpanInterceptor.

Translates spans emitted by any community OpenInference instrumentor
(``openinference-instrumentation-google-adk``, ``-openai``, ``-langchain``,
etc.) into ``confident.*`` OTel attrs that ``ConfidentSpanExporter`` rebuilds
into deepeval ``BaseSpan``s.

Mirrors the Pydantic AI POC pattern (and AgentCore's port of it): pushes
``BaseSpan`` placeholders for ``update_current_span(...)``, an implicit
``Trace`` placeholder (``_is_otel_implicit=True``) for bare callers, consumes
``next_*_span(...)`` payloads at on_start, resolves trace attrs FRESH at
on_end so live ``update_current_trace(...)`` mutations win, and stashes
``BaseMetric`` instances when an evaluation is running.

OpenInference-specific extraction (``openinference.span.kind``,
``llm.input_messages.{idx}``, ``llm.output_messages.{idx}``, ``tool.name``,
``tool.parameters``, ``llm.token_count.*``) is framework-written and
bypasses the placeholder serializer.
"""

from __future__ import annotations

import contextvars
import json
import logging
from time import perf_counter
from typing import Any, Dict, List, Optional

from deepeval.integrations.otel_instrumentation.base_instrumentation import (
    BaseInstrumentationSettings,
)
from deepeval.integrations.otel_instrumentation.utils import (
    SpanProcessor,
    bridge_otel_root_to_deepeval_parent,
    finalize_span_placeholder,
    pop_implicit_trace_context,
    push_implicit_trace_context,
    serialize_trace_context_to_otel_attrs,
)
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    pop_pending_for,
)
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.utils import (
    set_span_attribute_post_end,
    to_hex_string,
)
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    SpanType,
    Trace,
    TraceSpanStatus,
    ToolCall,
)
from deepeval.tracing.integrations import Integration
from deepeval.tracing.utils import (
    infer_provider_from_model,
    normalize_span_provider_for_platform,
)
from deepeval.utils import serialize_to_json

logger = logging.getLogger(__name__)

init_clock_bridge()


# OpenInference span classification. Reads ``openinference.span.kind`` (set by
# every OpenInference instrumentor); returns ``None`` for non-OI spans so the
# interceptor leaves them alone.


def _get_span_kind(span) -> Optional[str]:
    attrs = (
        getattr(span, "attributes", None)
        or getattr(span, "_attributes", None)
        or {}
    )
    kind = str(attrs.get("openinference.span.kind", "")).upper()

    if not kind:
        return None

    if kind in ("AGENT", "CHAIN"):
        return SpanType.AGENT
    if kind == "LLM":
        return SpanType.LLM
    if kind == "TOOL":
        return SpanType.TOOL
    if kind == "RETRIEVER":
        return SpanType.RETRIEVER

    return "custom"


def _get_agent_name(span) -> Optional[str]:
    attrs = (
        getattr(span, "attributes", None)
        or getattr(span, "_attributes", None)
        or {}
    )
    return attrs.get("agent.name") or span.name or None


def _get_tool_name(span) -> Optional[str]:
    attrs = (
        getattr(span, "attributes", None)
        or getattr(span, "_attributes", None)
        or {}
    )
    return attrs.get("tool.name") or span.name or None


# Content / I/O extraction. Walks OpenInference's flattened
# ``llm.input_messages.{idx}.message.*`` / ``llm.output_messages.{idx}...``
# semconv attrs (and the unflattened JSON-blob fallback) plus generic
# ``input.value`` / ``output.value`` for non-LLM spans.


def _extract_messages(span) -> tuple[Optional[str], Optional[str]]:
    attrs = (
        getattr(span, "attributes", None)
        or getattr(span, "_attributes", None)
        or {}
    )

    input_text = None
    output_text = None

    # 1. INPUT — flattened llm.input_messages.{idx}.message.content
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

    # Generic fallback (Agent / Tool / Chain spans)
    if not input_text:
        input_text = attrs.get("input.value")

    # 2. OUTPUT — symmetric to input
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

    if not output_text:
        output_text = attrs.get("output.value")

    return (
        str(input_text) if input_text is not None else None,
        str(output_text) if output_text is not None else None,
    )


def _extract_tool_calls(span) -> List[ToolCall]:
    """Tool calls embedded inside an LLM span's flattened output messages.

    Scenario A (the span itself is a tool span) is handled separately by
    ``_extract_tool_call_from_tool_span``; this helper covers Scenario B
    only — tool calls nested under ``llm.output_messages.{idx}.message
    .tool_calls.{tc_idx}.tool_call.function``.
    """
    attrs = (
        getattr(span, "attributes", None)
        or getattr(span, "_attributes", None)
        or {}
    )
    tools: List[ToolCall] = []

    msg_idx = 0
    while True:
        if (
            f"llm.output_messages.{msg_idx}.message.role" not in attrs
            and f"llm.output_messages.{msg_idx}.message.content" not in attrs
        ):
            break

        tc_idx = 0
        while True:
            base_key = (
                f"llm.output_messages.{msg_idx}.message.tool_calls."
                f"{tc_idx}.tool_call.function"
            )
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

    # Fallback: unflattened JSON blob.
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
                                    name=str(t_name),
                                    input_parameters=t_params,
                                )
                            )
        except Exception:
            pass

    return tools


def _extract_tool_call_from_tool_span(span) -> Optional[ToolCall]:
    tool_name = _get_tool_name(span)
    if not tool_name:
        return None

    attrs = (
        getattr(span, "attributes", None)
        or getattr(span, "_attributes", None)
        or {}
    )
    args_raw = attrs.get("tool.parameters") or attrs.get("input.value") or "{}"
    try:
        input_params = (
            json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        )
    except Exception:
        input_params = {}

    return ToolCall(name=tool_name, input_parameters=input_params)


# Settings: trace-level kwargs only. Span-level config goes on
# ``next_*_span(...)`` / ``update_current_span(...)`` — see README.


class OpenInferenceInstrumentationSettings(BaseInstrumentationSettings):
    """Trace-level defaults for OpenInference instrumentation.

    See ``BaseInstrumentationSettings`` for the accepted kwargs. ``integration``
    is overridable here because Google ADK reuses this interceptor and labels
    its spans with its own ``Integration`` value.
    """

    DEFAULT_INTEGRATION = Integration.OPEN_INFERENCE.value


class OpenInferenceSpanInterceptor(SpanProcessor):

    def __init__(self, settings_instance: OpenInferenceInstrumentationSettings):
        self.settings = settings_instance
        # Per-OTel-span state keyed by span_id (unique within a process).
        self._tokens: Dict[int, contextvars.Token] = {}
        self._placeholders: Dict[int, BaseSpan] = {}
        # Implicit-trace state, keyed on the OTel root span_id that pushed it.
        self._trace_tokens: Dict[int, contextvars.Token] = {}
        self._trace_placeholders: Dict[int, Trace] = {}

    def on_start(self, span, parent_context):
        # Order matches Pydantic AI: implicit-trace push before classification
        # so anything reading ``current_trace_context`` downstream sees it.
        push_implicit_trace_context(
            span, self._trace_tokens, self._trace_placeholders
        )
        bridge_otel_root_to_deepeval_parent(span)

        span_type = _get_span_kind(span)
        if span_type:
            try:
                span.set_attribute(ConfidentAttr.SPAN_TYPE, span_type)
                span.set_attribute(
                    ConfidentAttr.SPAN_INTEGRATION, self.settings.integration
                )
            except Exception:
                pass

        # Stamp name at on_start because the placeholder subclass depends on it.
        if span_type == SpanType.AGENT:
            agent_name = _get_agent_name(span)
            if agent_name:
                try:
                    span.set_attribute(ConfidentAttr.SPAN_NAME, agent_name)
                except Exception:
                    pass
        elif span_type == SpanType.TOOL:
            tool_name = _get_tool_name(span)
            if tool_name:
                try:
                    span.set_attribute(ConfidentAttr.SPAN_NAME, tool_name)
                except Exception:
                    pass

        self._push_span_context(span, span_type)

    def on_end(self, span):
        sid = span.get_span_context().span_id

        # Resolve trace attrs FRESH so live ``update_current_trace(...)`` wins.
        try:
            serialize_trace_context_to_otel_attrs(span, self.settings)
        except Exception as exc:
            logger.debug(
                "Failed to serialize trace context for span_id=%s: %s",
                sid,
                exc,
            )

        finalize_span_placeholder(span, self._tokens, self._placeholders)

        # Framework attrs are non-user-mutable; written alongside (not inside)
        # the placeholder serializer.
        try:
            self._serialize_framework_attrs(span)
        except Exception as exc:
            logger.debug(
                "Failed to serialize framework attrs for span_id=%s: %s",
                sid,
                exc,
            )

        # Must run AFTER trace serialization so the implicit placeholder's
        # mutations land on this root's attrs.
        pop_implicit_trace_context(
            span, self._trace_tokens, self._trace_placeholders
        )

    def _push_span_context(self, span, span_type: Optional[str]) -> None:
        """Push a typed placeholder span onto the contextvar.

        Consumes ``next_*_span(...)`` defaults BEFORE the push so user code
        sees the staged values.
        """
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            start_time = (
                peb.epoch_nanos_to_perf_seconds(span.start_time)
                if span.start_time
                else perf_counter()
            )
            kwargs: Dict[str, Any] = dict(
                uuid=to_hex_string(sid, 16),
                trace_uuid=to_hex_string(tid, 32),
                status=TraceSpanStatus.IN_PROGRESS,
                start_time=start_time,
            )
            if span_type == SpanType.AGENT:
                # Reuse the on_start-stamped name to skip a duplicate lookup.
                attrs = (
                    getattr(span, "attributes", None)
                    or getattr(span, "_attributes", None)
                    or {}
                )
                placeholder = AgentSpan(
                    name=(
                        attrs.get(ConfidentAttr.SPAN_NAME)
                        or _get_agent_name(span)
                        or "agent"
                    ),
                    **kwargs,
                )
            elif span_type == SpanType.LLM:
                placeholder = LlmSpan(**kwargs)
            else:
                placeholder = BaseSpan(**kwargs)

            pending = pop_pending_for(span_type)
            if pending:
                apply_pending_to_span(placeholder, pending)

            token = current_span_context.set(placeholder)
            self._tokens[sid] = token
            self._placeholders[sid] = placeholder
        except Exception as exc:
            logger.debug(
                "Failed to push current_span_context placeholder: %s", exc
            )

    def _serialize_framework_attrs(self, span) -> None:
        """Translate OpenInference attrs into ``confident.*``.

        Uses ``setdefault`` semantics — the placeholder serializer ran first,
        so user mutations win.
        """
        attrs = (
            getattr(span, "attributes", None)
            or getattr(span, "_attributes", None)
            or {}
        )
        span_type = attrs.get(ConfidentAttr.SPAN_TYPE) or _get_span_kind(span)
        if span_type and ConfidentAttr.SPAN_TYPE not in attrs:
            set_span_attribute_post_end(
                span, ConfidentAttr.SPAN_TYPE, span_type
            )
        if (
            self.settings.integration
            and ConfidentAttr.SPAN_INTEGRATION not in attrs
        ):
            set_span_attribute_post_end(
                span,
                ConfidentAttr.SPAN_INTEGRATION,
                self.settings.integration,
            )

        input_text, output_text = _extract_messages(span)

        if input_text and ConfidentAttr.SPAN_INPUT not in attrs:
            set_span_attribute_post_end(
                span, ConfidentAttr.SPAN_INPUT, input_text
            )
            if span_type == SpanType.AGENT:
                set_span_attribute_post_end(
                    span, ConfidentAttr.TRACE_INPUT, input_text
                )

        if output_text and ConfidentAttr.SPAN_OUTPUT not in attrs:
            set_span_attribute_post_end(
                span, ConfidentAttr.SPAN_OUTPUT, output_text
            )
            if span_type == SpanType.AGENT:
                set_span_attribute_post_end(
                    span, ConfidentAttr.TRACE_OUTPUT, output_text
                )

        # Token usage — OpenInference uses ``llm.token_count.{prompt,completion}``.
        input_tokens = attrs.get("llm.token_count.prompt")
        output_tokens = attrs.get("llm.token_count.completion")
        if input_tokens is not None and not attrs.get(
            ConfidentAttr.LLM_INPUT_TOKEN_COUNT
        ):
            set_span_attribute_post_end(
                span, ConfidentAttr.LLM_INPUT_TOKEN_COUNT, int(input_tokens)
            )
        if output_tokens is not None and not attrs.get(
            ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT
        ):
            set_span_attribute_post_end(
                span, ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT, int(output_tokens)
            )

        model = attrs.get("llm.model_name")
        if model and not attrs.get(ConfidentAttr.LLM_MODEL):
            set_span_attribute_post_end(
                span, ConfidentAttr.LLM_MODEL, str(model)
            )
        if span_type == SpanType.LLM and not attrs.get(
            ConfidentAttr.SPAN_PROVIDER
        ):
            provider = attrs.get("llm.provider")
            if not provider and model:
                provider = infer_provider_from_model(str(model))
            if provider:
                provider = normalize_span_provider_for_platform(provider)
                set_span_attribute_post_end(
                    span, ConfidentAttr.SPAN_PROVIDER, str(provider)
                )

        tools_called: List[ToolCall] = []

        if span_type == SpanType.TOOL:
            tc = _extract_tool_call_from_tool_span(span)
            if tc:
                tools_called = [tc]

                if (
                    tc.input_parameters
                    and ConfidentAttr.SPAN_INPUT not in attrs
                ):
                    set_span_attribute_post_end(
                        span,
                        ConfidentAttr.SPAN_INPUT,
                        serialize_to_json(tc.input_parameters),
                    )

        elif span_type in (SpanType.AGENT, SpanType.LLM):
            tools_called = _extract_tool_calls(span)

        if tools_called:
            set_span_attribute_post_end(
                span,
                ConfidentAttr.SPAN_TOOLS_CALLED,
                [t.model_dump_json() for t in tools_called],
            )

        if (
            span_type == SpanType.AGENT
            and ConfidentAttr.SPAN_NAME not in attrs
        ):
            agent_name = _get_agent_name(span)
            if agent_name:
                set_span_attribute_post_end(
                    span, ConfidentAttr.SPAN_NAME, agent_name
                )
