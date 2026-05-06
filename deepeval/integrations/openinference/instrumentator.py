"""OpenInference × deepeval OTel SpanInterceptor.

Translates spans emitted by any community OpenInference instrumentor
(``openinference-instrumentation-google-adk``, ``-openai``, ``-langchain``,
etc.) into ``confident.*`` OTel attrs that ``ConfidentSpanExporter`` rebuilds
into deepeval ``BaseSpan``s.

Mirrors the Pydantic AI POC pattern (and AgentCore's port of it): pushes
``BaseSpan`` placeholders for ``update_current_span(...)``, an implicit
``Trace(is_otel_implicit=True)`` for bare callers, consumes
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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    current_trace_context,
    pop_pending_for,
)
from deepeval.tracing.otel.utils import (
    stash_pending_metrics,
    to_hex_string,
)
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
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
        return "agent"
    if kind == "LLM":
        return "llm"
    if kind == "TOOL":
        return "tool"
    if kind == "RETRIEVER":
        return "retriever"

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


class OpenInferenceInstrumentationSettings:
    """Trace-level defaults for OpenInference instrumentation.

    All kwargs are optional. Trace fields are resolved at every span's
    ``on_end`` so runtime ``update_current_trace(...)`` mutations win.
    ``api_key`` is optional; when omitted, the OTel pipeline runs locally
    but the Confident AI backend rejects uploads.
    """

    # Span-level kwargs removed in the OTel POC migration — raise on use.
    _REMOVED_KWARGS = (
        "is_test_mode",
        "agent_metric_collection",
        "llm_metric_collection",
        "tool_metric_collection_map",
        "trace_metric_collection",
        "agent_metrics",
        "confident_prompt",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        metric_collection: Optional[str] = None,
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        environment: Optional[str] = None,
        **removed_kwargs: Any,
    ):
        is_dependency_installed()

        # ``**removed_kwargs`` exists only to produce a crisp migration error.
        if removed_kwargs:
            offending = ", ".join(sorted(removed_kwargs))
            raise TypeError(
                f"OpenInferenceInstrumentationSettings: unexpected keyword "
                f"argument(s) {offending}. Span-level kwargs were removed "
                "in the OTel POC migration; use ``with next_*_span(...)`` "
                "or ``update_current_span(...)``. "
                "See deepeval/integrations/README.md."
            )

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

        self.api_key = api_key
        self.name = name
        self.thread_id = thread_id
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.metric_collection = metric_collection
        self.test_case_id = test_case_id
        self.turn_id = turn_id


# Span interceptor. Pushes BaseSpan placeholders for ``update_current_span``,
# implicit Trace for bare callers, parent-uuid bridge for OTel roots inside
# ``@observe``, ``next_*_span`` consumption, and framework-attr extraction.


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
        self._maybe_push_implicit_trace_context(span)
        self._maybe_bridge_otel_root_to_deepeval_parent(span)

        span_type = _get_span_kind(span)
        if span_type:
            try:
                span.set_attribute("confident.span.type", span_type)
            except Exception:
                pass

        # Stamp name at on_start because the placeholder subclass depends on it.
        if span_type == "agent":
            agent_name = _get_agent_name(span)
            if agent_name:
                try:
                    span.set_attribute("confident.span.name", agent_name)
                except Exception:
                    pass
        elif span_type == "tool":
            tool_name = _get_tool_name(span)
            if tool_name:
                try:
                    span.set_attribute("confident.span.name", tool_name)
                except Exception:
                    pass

        self._push_span_context(span, span_type)

    def on_end(self, span):
        sid = span.get_span_context().span_id

        # Resolve trace attrs FRESH so live ``update_current_trace(...)`` wins.
        try:
            self._serialize_trace_context_to_otel_attrs(span)
        except Exception as exc:
            logger.debug(
                "Failed to serialize trace context for span_id=%s: %s",
                sid,
                exc,
            )

        placeholder = self._placeholders.pop(sid, None)
        token = self._tokens.pop(sid, None)
        if token is not None:
            try:
                current_span_context.reset(token)
            except Exception as exc:
                logger.debug(
                    "Failed to reset current_span_context for span_id=%s: %s",
                    sid,
                    exc,
                )
        if placeholder is not None:
            try:
                self._serialize_placeholder_to_otel_attrs(placeholder, span)
            except Exception as exc:
                logger.debug(
                    "Failed to serialize span placeholder for span_id=%s: %s",
                    sid,
                    exc,
                )
            try:
                if placeholder.metrics and trace_manager.is_evaluating:
                    stash_pending_metrics(
                        to_hex_string(sid, 16), placeholder.metrics
                    )
            except Exception as exc:
                logger.debug(
                    "Failed to stash pending metrics for span_id=%s: %s",
                    sid,
                    exc,
                )

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
        self._maybe_pop_implicit_trace_context(span)

    def _push_span_context(self, span, span_type: Optional[str]) -> None:
        """Push a ``BaseSpan`` / ``AgentSpan`` placeholder onto the contextvar.

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
            if span_type == "agent":
                # Reuse the on_start-stamped name to skip a duplicate lookup.
                attrs = (
                    getattr(span, "attributes", None)
                    or getattr(span, "_attributes", None)
                    or {}
                )
                placeholder = AgentSpan(
                    name=(
                        attrs.get("confident.span.name")
                        or _get_agent_name(span)
                        or "agent"
                    ),
                    **kwargs,
                )
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

    def _maybe_push_implicit_trace_context(self, span) -> None:
        """Push an implicit ``Trace`` for OTel roots without enclosing context.

        Tagged ``is_otel_implicit=True`` so ``ContextAwareSpanProcessor``
        still routes to OTLP.
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
                is_otel_implicit=True,
            )
            token = current_trace_context.set(implicit)
            self._trace_tokens[sid] = token
            self._trace_placeholders[sid] = implicit
        except Exception as exc:
            logger.debug(
                "Failed to push implicit current_trace_context: %s", exc
            )

    def _maybe_bridge_otel_root_to_deepeval_parent(self, span) -> None:
        """Re-parent OTel roots onto an enclosing ``@observe`` deepeval span.

        Stamps ``confident.span.parent_uuid`` so the exporter stitches the
        OTel root into the deepeval parent's trace instead of leaving them
        as siblings.
        """
        if getattr(span, "parent", None) is not None:
            return
        parent_span = current_span_context.get()
        if parent_span is None:
            return
        parent_uuid = getattr(parent_span, "uuid", None)
        if not parent_uuid:
            return
        try:
            self._set_attr_post_end(
                span, "confident.span.parent_uuid", parent_uuid
            )
        except Exception as exc:
            logger.debug(
                "Failed to bridge OTel root span to deepeval parent "
                "(parent_uuid=%s): %s",
                parent_uuid,
                exc,
            )

    def _maybe_pop_implicit_trace_context(self, span) -> None:
        try:
            sid = span.get_span_context().span_id
        except Exception:
            return
        token = self._trace_tokens.pop(sid, None)
        self._trace_placeholders.pop(sid, None)
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

        ``Span.set_attribute`` is a no-op after ``Span.end()``, so we write
        directly through ``_attributes`` (mutable while processors are
        running) and fall back to ``set_attribute`` if that fails.
        """
        try:
            attrs = getattr(span, "_attributes", None)
            if attrs is not None:
                attrs[key] = value
                return
        except Exception as exc:
            logger.debug(
                "Direct _attributes write failed for %s; "
                "falling back to set_attribute (may be dropped): %s",
                key,
                exc,
            )
        try:
            span.set_attribute(key, value)
        except Exception as exc:
            logger.debug("set_attribute fallback failed for %s: %s", key, exc)

    @classmethod
    def _serialize_placeholder_to_otel_attrs(
        cls, placeholder: BaseSpan, span
    ) -> None:
        """Mirror ``update_current_span`` writes onto ``confident.span.*``.

        Only writes user-set fields; doesn't overwrite on_start-stamped attrs.
        """
        existing = (
            getattr(span, "attributes", None)
            or getattr(span, "_attributes", None)
            or {}
        )

        if placeholder.metadata:
            cls._set_attr_post_end(
                span,
                "confident.span.metadata",
                json.dumps(placeholder.metadata, default=str),
            )
        if placeholder.input is not None:
            cls._set_attr_post_end(
                span,
                "confident.span.input",
                json.dumps(placeholder.input, default=str),
            )
        if placeholder.output is not None:
            cls._set_attr_post_end(
                span,
                "confident.span.output",
                json.dumps(placeholder.output, default=str),
            )
        if placeholder.metric_collection:
            cls._set_attr_post_end(
                span,
                "confident.span.metric_collection",
                placeholder.metric_collection,
            )
        if placeholder.retrieval_context:
            cls._set_attr_post_end(
                span,
                "confident.span.retrieval_context",
                json.dumps(placeholder.retrieval_context),
            )
        if placeholder.context:
            cls._set_attr_post_end(
                span,
                "confident.span.context",
                json.dumps(placeholder.context),
            )
        if placeholder.expected_output:
            cls._set_attr_post_end(
                span,
                "confident.span.expected_output",
                placeholder.expected_output,
            )
        if placeholder.name and not existing.get("confident.span.name"):
            cls._set_attr_post_end(
                span, "confident.span.name", placeholder.name
            )

    def _serialize_trace_context_to_otel_attrs(self, span) -> None:
        """Resolve trace attrs FRESH and write to ``confident.trace.*``.

        Reads ``current_trace_context.get()`` (so live
        ``update_current_trace(...)`` mutations win) with
        ``self.settings.*`` as fallback. Metadata is settings-base merged
        with runtime context on top.
        """
        trace_ctx = current_trace_context.get()

        _name = (trace_ctx.name if trace_ctx else None) or self.settings.name
        _thread_id = (
            trace_ctx.thread_id if trace_ctx else None
        ) or self.settings.thread_id
        _user_id = (
            trace_ctx.user_id if trace_ctx else None
        ) or self.settings.user_id
        _tags = (trace_ctx.tags if trace_ctx else None) or self.settings.tags
        _test_case_id = (
            trace_ctx.test_case_id if trace_ctx else None
        ) or self.settings.test_case_id
        _turn_id = (
            trace_ctx.turn_id if trace_ctx else None
        ) or self.settings.turn_id
        _trace_metric_collection = (
            trace_ctx.metric_collection if trace_ctx else None
        ) or self.settings.metric_collection
        _metadata = {
            **(self.settings.metadata or {}),
            **((trace_ctx.metadata or {}) if trace_ctx else {}),
        }

        if _name:
            self._set_attr_post_end(span, "confident.trace.name", _name)
        if _thread_id:
            self._set_attr_post_end(
                span, "confident.trace.thread_id", _thread_id
            )
        if _user_id:
            self._set_attr_post_end(span, "confident.trace.user_id", _user_id)
        if _tags:
            self._set_attr_post_end(span, "confident.trace.tags", _tags)
        if _metadata:
            self._set_attr_post_end(
                span, "confident.trace.metadata", json.dumps(_metadata)
            )
        if _trace_metric_collection:
            self._set_attr_post_end(
                span,
                "confident.trace.metric_collection",
                _trace_metric_collection,
            )
        if _test_case_id:
            self._set_attr_post_end(
                span, "confident.trace.test_case_id", _test_case_id
            )
        if _turn_id:
            self._set_attr_post_end(span, "confident.trace.turn_id", _turn_id)
        if self.settings.environment:
            self._set_attr_post_end(
                span,
                "confident.trace.environment",
                self.settings.environment,
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
        span_type = attrs.get("confident.span.type") or _get_span_kind(span)
        if span_type and "confident.span.type" not in attrs:
            self._set_attr_post_end(span, "confident.span.type", span_type)

        input_text, output_text = _extract_messages(span)

        if input_text and "confident.span.input" not in attrs:
            self._set_attr_post_end(span, "confident.span.input", input_text)
            if span_type == "agent":
                self._set_attr_post_end(
                    span, "confident.trace.input", input_text
                )

        if output_text and "confident.span.output" not in attrs:
            self._set_attr_post_end(span, "confident.span.output", output_text)
            if span_type == "agent":
                self._set_attr_post_end(
                    span, "confident.trace.output", output_text
                )

        # Token usage — OpenInference uses ``llm.token_count.{prompt,completion}``.
        input_tokens = attrs.get("llm.token_count.prompt")
        output_tokens = attrs.get("llm.token_count.completion")
        if input_tokens is not None:
            self._set_attr_post_end(
                span, "confident.llm.input_token_count", int(input_tokens)
            )
        if output_tokens is not None:
            self._set_attr_post_end(
                span, "confident.llm.output_token_count", int(output_tokens)
            )

        model = attrs.get("llm.model_name")
        if model:
            self._set_attr_post_end(span, "confident.llm.model", str(model))

        tools_called: List[ToolCall] = []

        if span_type == "tool":
            tc = _extract_tool_call_from_tool_span(span)
            if tc:
                tools_called = [tc]

                if tc.input_parameters and "confident.span.input" not in attrs:
                    self._set_attr_post_end(
                        span,
                        "confident.span.input",
                        json.dumps(tc.input_parameters),
                    )

        elif span_type in ("agent", "llm"):
            tools_called = _extract_tool_calls(span)

        if tools_called:
            self._set_attr_post_end(
                span,
                "confident.span.tools_called",
                [t.model_dump_json() for t in tools_called],
            )

        if span_type == "agent" and "confident.span.name" not in attrs:
            agent_name = _get_agent_name(span)
            if agent_name:
                self._set_attr_post_end(span, "confident.span.name", agent_name)
