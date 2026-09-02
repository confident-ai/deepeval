"""AgentCore × deepeval OTel SpanInterceptor.

Translates AWS Bedrock AgentCore / Strands / Traceloop spans into
``confident.*`` OTel attrs that ``ConfidentSpanExporter`` rebuilds into
deepeval ``BaseSpan``s. Mirrors the Pydantic AI POC pattern: pushes
``BaseSpan`` placeholders for ``update_current_span(...)``, an implicit
``Trace`` placeholder (``_is_otel_implicit=True``) for bare callers, consumes
``next_*_span(...)`` payloads at on_start, resolves trace attrs FRESH
at on_end, and stashes ``BaseMetric`` instances when evaluating.

The framework-agnostic half of that pattern lives in
``deepeval.integrations.otel_instrumentation.utils``; the ``gen_ai.*`` /
Traceloop attribute readers live in the sibling ``genai_attributes`` module.
Only AgentCore-specific extraction (AWS Bedrock body parsing) stays here, and
it bypasses the placeholder serializer.
"""

from __future__ import annotations

import contextvars
import logging
from time import perf_counter
from typing import Any, Dict, List, Optional

from deepeval.integrations.otel_instrumentation.genai_attributes import (
    classify_span,
    extract_messages,
    extract_tool_call_from_tool_span,
    extract_tool_calls,
    get_agent_name,
    get_attr,
    get_tool_name,
)
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
from deepeval.tracing.integrations import Integration
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
from deepeval.tracing.utils import (
    infer_provider_from_model,
    normalize_span_provider_for_platform,
)
from deepeval.utils import serialize_to_json

logger = logging.getLogger(__name__)

init_clock_bridge()


class AgentCoreInstrumentationSettings(BaseInstrumentationSettings):
    """Trace-level defaults for AgentCore instrumentation.

    See ``BaseInstrumentationSettings`` for the accepted kwargs.
    """

    DEFAULT_INTEGRATION = Integration.AGENTCORE.value


class AgentCoreSpanInterceptor(SpanProcessor):

    def __init__(self, settings_instance: AgentCoreInstrumentationSettings):
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

        span_type = classify_span(span)
        if span_type:
            try:
                span.set_attribute(ConfidentAttr.SPAN_TYPE, span_type)
            except Exception:
                pass

        # Stamp name at on_start because the placeholder subclass depends on it.
        if span_type == SpanType.AGENT:
            agent_name = get_agent_name(span)
            if agent_name:
                try:
                    span.set_attribute(ConfidentAttr.SPAN_NAME, agent_name)
                except Exception:
                    pass
        elif span_type == SpanType.TOOL:
            tool_name = get_tool_name(span)
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
            serialize_trace_context_to_otel_attrs(
                span, self.settings, thread_id_fallback_attr="session.id"
            )
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
                attrs = span.attributes or {}
                placeholder = AgentSpan(
                    name=(
                        attrs.get(ConfidentAttr.SPAN_NAME)
                        or get_agent_name(span)
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
        """Translate Strands / Traceloop / GenAI attrs into ``confident.*``.

        Uses ``setdefault`` semantics — the placeholder serializer ran first,
        so user mutations win.
        """
        attrs = span.attributes or {}
        span_type = attrs.get(ConfidentAttr.SPAN_TYPE) or classify_span(span)
        if span_type and ConfidentAttr.SPAN_TYPE not in attrs:
            set_span_attribute_post_end(
                span, ConfidentAttr.SPAN_TYPE, span_type
            )
        if not attrs.get(ConfidentAttr.SPAN_INTEGRATION):
            set_span_attribute_post_end(
                span,
                ConfidentAttr.SPAN_INTEGRATION,
                Integration.AGENTCORE.value,
            )

        input_text, output_text = extract_messages(span)

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

        input_tokens = attrs.get("gen_ai.usage.input_tokens") or attrs.get(
            "gen_ai.usage.prompt_tokens"
        )
        output_tokens = attrs.get("gen_ai.usage.output_tokens") or attrs.get(
            "gen_ai.usage.completion_tokens"
        )
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

        model = get_attr(
            span,
            "gen_ai.response.model",
            "gen_ai.request.model",
        )
        if model:
            if not attrs.get(ConfidentAttr.LLM_MODEL):
                set_span_attribute_post_end(
                    span, ConfidentAttr.LLM_MODEL, model
                )
            if span_type == SpanType.LLM and not attrs.get(
                ConfidentAttr.SPAN_PROVIDER
            ):
                provider = infer_provider_from_model(model)
                if provider:
                    provider = normalize_span_provider_for_platform(provider)
                    set_span_attribute_post_end(
                        span, ConfidentAttr.SPAN_PROVIDER, provider
                    )

        tools_called: List[ToolCall] = []

        if span_type == SpanType.AGENT:
            tools_called = extract_tool_calls(span)

            tool_defs_raw = attrs.get("gen_ai.tool.definitions") or attrs.get(
                "gen_ai.agent.tools"
            )
            if tool_defs_raw:
                set_span_attribute_post_end(
                    span,
                    ConfidentAttr.AGENT_TOOL_DEFINITIONS,
                    str(tool_defs_raw),
                )

        elif span_type == SpanType.TOOL:
            tc = extract_tool_call_from_tool_span(span)
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

            if ConfidentAttr.SPAN_OUTPUT not in attrs:
                raw_output = get_attr(
                    span, "traceloop.entity.output", "gen_ai.tool.output"
                )
                if raw_output:
                    set_span_attribute_post_end(
                        span, ConfidentAttr.SPAN_OUTPUT, raw_output
                    )

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
            agent_name = get_agent_name(span)
            if agent_name:
                set_span_attribute_post_end(
                    span, ConfidentAttr.SPAN_NAME, agent_name
                )
