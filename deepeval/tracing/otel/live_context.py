"""Shared live context support for native and confident-trace OTEL spans."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Dict, Optional

from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    pop_pending_for,
    apply_pending_to_span,
)
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    ToolSpan,
    RetrieverSpan,
    SpanType,
    Trace,
    TraceSpanStatus,
)
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.utils import (
    to_hex_string,
    set_span_attribute_post_end,
)

logger = logging.getLogger(__name__)


class LiveSpanContext:
    @staticmethod
    def _span_key(span):
        context = span.get_span_context()
        return context.trace_id, context.span_id

    def _push_span_context(self, span, span_type: Optional[str]) -> None:
        """Push a typed placeholder span onto the contextvar.

        Consumes ``next_*_span(...)`` defaults BEFORE the push so user code
        sees the staged values.
        """
        try:
            if not span_type:
                from deepeval.tracing.otel.utils import (
                    check_span_type_from_gen_ai_attributes,
                )

                span_type = check_span_type_from_gen_ai_attributes(span)
                if span_type != "base":
                    span.set_attribute(ConfidentAttr.SPAN_TYPE, span_type)
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
            if span_type == SpanType.AGENT.value:
                # Reuse the on_start-stamped name to skip a duplicate lookup.
                attrs = span.attributes or {}
                placeholder = AgentSpan(
                    name=(
                        attrs.get(ConfidentAttr.SPAN_NAME)
                        or (
                            attrs.get("gen_ai.agent.name")
                            or attrs.get("agent_name")
                        )
                        or "agent"
                    ),
                    **kwargs,
                )
            elif span_type == SpanType.LLM.value:
                placeholder = LlmSpan(**kwargs)
            elif span_type == SpanType.TOOL.value:
                placeholder = ToolSpan(
                    name=(span.attributes or {}).get(ConfidentAttr.SPAN_NAME)
                    or (span.attributes or {}).get("gen_ai.tool.name")
                    or (span.attributes or {}).get("tool.name")
                    or span.name
                    or "tool",
                    **kwargs,
                )
            elif span_type == SpanType.RETRIEVER.value:
                placeholder = RetrieverSpan(embedder="", **kwargs)
            else:
                placeholder = BaseSpan(**kwargs)

            pending = pop_pending_for(span_type)
            if pending:
                apply_pending_to_span(placeholder, pending)

            placeholder._otel_parent = current_span_context.get()
            token = current_span_context.set(placeholder)
            self._tokens[self._span_key(span)] = token
            self._placeholders[self._span_key(span)] = placeholder
        except Exception as exc:
            logger.debug(
                "Failed to push current_span_context placeholder: %s", exc
            )

    def _maybe_push_implicit_trace_context(self, span) -> None:
        """Push an implicit ``Trace`` for OTel roots without enclosing context.

        Tagged ``_is_otel_implicit=True`` so ``ContextAwareSpanProcessor``
        still routes to OTLP. ``_is_otel_implicit`` is a Pydantic
        ``PrivateAttr``, so it must be set after construction (it's not a
        constructor kwarg).
        """
        if current_trace_context.get() is not None:
            return
        if getattr(span, "parent", None) is not None:
            return
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            start_time = (
                peb.epoch_nanos_to_perf_seconds(span.start_time)
                if span.start_time
                else perf_counter()
            )
            implicit = Trace(
                uuid=to_hex_string(tid, 32),
                root_spans=[],
                status=TraceSpanStatus.IN_PROGRESS,
                start_time=start_time,
            )
            implicit._is_otel_implicit = True
            token = current_trace_context.set(implicit)
            self._trace_tokens[self._span_key(span)] = token
            self._trace_placeholders[self._span_key(span)] = implicit
        except Exception as exc:
            logger.debug(
                "Failed to push implicit current_trace_context: %s", exc
            )

    def _maybe_pop_implicit_trace_context(self, span) -> None:
        try:
            sid = span.get_span_context().span_id
        except Exception:
            return
        token = self._trace_tokens.pop(self._span_key(span), None)
        self._trace_placeholders.pop(self._span_key(span), None)
        if token is None:
            return
        try:
            current_trace_context.reset(token)
        except Exception as exc:
            logger.debug(
                "Failed to reset implicit current_trace_context for "
                "span_id=%s: %s",
                sid,
                exc,
            )

    @staticmethod
    def _set_attr_post_end(span, key: str, value: Any) -> None:
        """Write to a span that may have ended.

        ``Span.set_attribute`` is a no-op after ``Span.end()`` and ``on_end``
        receives a ``ReadableSpan`` that has no such method, so the write goes
        through the span's ``_attributes`` mapping — see
        ``set_span_attribute_post_end``.
        """
        set_span_attribute_post_end(span, key, value)

    def discard_context(self, key):
        """Release unfinished placeholders without resetting another task's context."""
        placeholder = self._placeholders.pop(key, None)
        if placeholder is not None and placeholder.end_time is None:
            placeholder.end_time = perf_counter()
        token = self._tokens.pop(key, None)
        if token is not None and current_span_context.get() is placeholder:
            try:
                current_span_context.reset(token)
            except ValueError:
                current_span_context.set(None)
        trace = self._trace_placeholders.pop(key, None)
        token = self._trace_tokens.pop(key, None)
        if token is not None and current_trace_context.get() is trace:
            try:
                current_trace_context.reset(token)
            except ValueError:
                current_trace_context.set(None)
