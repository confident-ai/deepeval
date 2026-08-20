"""Runtime helpers shared by the OTel integrations' ``SpanInterceptor``s.

``openinference``, ``pydantic_ai``, ``agentcore`` and ``strands`` all
implement the same pattern: push ``BaseSpan`` / ``Trace`` placeholders onto
deepeval's contextvars at ``on_start`` so ``update_current_span(...)`` /
``update_current_trace(...)`` have something to mutate from anywhere in the
call stack, then serialize those placeholders back into ``confident.*`` OTel
attributes at ``on_end``.

This module owns the framework-agnostic half of that lifecycle, plus the
optional-import shim for the OTel SDK. The ``instrument_*()`` setup path —
settings base class and ``TracerProvider`` registration — lives in the
sibling ``base_instrumentation`` module; span classification and framework
attribute extraction stay in each integration's own ``instrumentator.py``.
"""

from __future__ import annotations

import contextvars
import logging
from time import perf_counter
from typing import Any, Dict, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.utils import (
    serialize_placeholder_to_otel_attrs,
    set_span_attribute_post_end,
    stash_pending_metrics,
    to_hex_string,
)
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, Trace, TraceSpanStatus
from deepeval.utils import serialize_to_json

logger = logging.getLogger(__name__)
settings = get_settings()


OTEL_INSTALL_HINT = (
    "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
)


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


if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
else:
    SpanProcessor = _SpanProcessor
    ReadableSpan = _ReadableSpan


def is_dependency_installed(install_hint: str = OTEL_INSTALL_HINT) -> bool:
    """Raise a install-hint ``ImportError`` when the OTel SDK is missing."""
    if not dependency_installed:
        raise ImportError(
            f"Dependencies are not installed. Please install them with "
            f"`{install_hint}`."
        )
    return True


# Span / trace context placeholders. The interceptor owns the state dicts
# (keyed by OTel span_id, unique within a process) and passes them in, so
# these stay plain functions rather than forcing a shared base class onto
# four interceptors whose on_start / on_end orchestration differs.


def push_implicit_trace_context(
    span,
    trace_tokens: Dict[int, contextvars.Token],
    trace_placeholders: Dict[int, Trace],
) -> None:
    """Push an implicit ``Trace`` for OTel roots without enclosing context.

    Only fires for the OTel root span, and only when the caller hasn't
    already pushed their own trace context (via ``@observe`` /
    ``with trace(...)``). The placeholder exists so that
    ``update_current_trace(...)`` from inside tools / nested helpers has a
    target to mutate.

    Tagged ``_is_otel_implicit=True`` so ``ContextAwareSpanProcessor`` knows
    NOT to switch routing to REST — bare callers expect OTLP.
    ``_is_otel_implicit`` is a Pydantic ``PrivateAttr``, so it must be set
    after construction (it's not a constructor kwarg).
    """
    if current_trace_context.get() is not None:
        return  # user already owns the trace context; don't touch it
    # Child spans inherit the placeholder via contextvars and never need
    # their own.
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
        trace_tokens[sid] = token
        trace_placeholders[sid] = implicit
    except Exception as exc:
        logger.debug("Failed to push implicit current_trace_context: %s", exc)


def pop_implicit_trace_context(
    span,
    trace_tokens: Dict[int, contextvars.Token],
    trace_placeholders: Dict[int, Trace],
) -> None:
    """Reset the implicit trace contextvar if this span is the one that pushed.

    Must run AFTER trace-attr serialization so the placeholder's mutations
    land on the root span's attrs.
    """
    try:
        sid = span.get_span_context().span_id
    except Exception:
        return
    token = trace_tokens.pop(sid, None)
    trace_placeholders.pop(sid, None)
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


def bridge_otel_root_to_deepeval_parent(
    span, integration: Optional[str] = None
) -> None:
    """Re-parent an OTel root span onto its enclosing deepeval span.

    When ``@observe`` (or any deepeval-managed span) wraps a bare framework
    call, the deepeval span is pushed onto ``current_span_context`` but no
    OTel parent context is established. The framework then opens an OTel
    root span, and the exporter would emit it as a second trace root sibling
    to the ``@observe`` span rather than as its child.

    Stamping ``confident.span.parent_uuid`` closes that gap:
    ``ConfidentSpanExporter`` prefers the override iff the OTel span has no
    native parent, so this never overrides a real parent_id.

    ``integration`` backfills the enclosing deepeval span's integration label
    when it has none; omit it to leave the parent untouched.
    """
    # Only OTel roots need bridging; child OTel spans already have a real
    # parent_id pointing into the same OTel trace.
    if getattr(span, "parent", None) is not None:
        return
    parent_span = current_span_context.get()
    if parent_span is None:
        return
    parent_uuid = getattr(parent_span, "uuid", None)
    if not parent_uuid:
        return
    if integration and not getattr(parent_span, "integration", None):
        try:
            parent_span.integration = integration
        except Exception:
            pass
    try:
        set_span_attribute_post_end(
            span, ConfidentAttr.SPAN_PARENT_UUID, parent_uuid
        )
    except Exception as exc:
        logger.debug(
            "Failed to bridge OTel root span to deepeval parent "
            "(parent_uuid=%s): %s",
            parent_uuid,
            exc,
        )


def finalize_span_placeholder(
    span,
    tokens: Dict[int, contextvars.Token],
    placeholders: Dict[int, BaseSpan],
) -> None:
    """Pop the span placeholder at ``on_end`` and serialize user mutations.

    Resets ``current_span_context``, writes the placeholder's mutated fields
    into ``confident.span.*`` attrs, and hands off any ``BaseMetric``
    instances — those can't ride in OTel attrs (primitives only), so the
    in-process overlay re-attaches them. The eval-mode gate keeps the
    registry from growing in prod paths where the OTLP collector lives in
    another process and the reader never fires.
    """
    sid = span.get_span_context().span_id
    placeholder = placeholders.pop(sid, None)
    token = tokens.pop(sid, None)
    if token is not None:
        try:
            current_span_context.reset(token)
        except Exception as exc:
            logger.debug(
                "Failed to reset current_span_context for span_id=%s: %s",
                sid,
                exc,
            )
    if placeholder is None:
        return
    try:
        serialize_placeholder_to_otel_attrs(span, placeholder)
    except Exception as exc:
        logger.debug(
            "Failed to serialize span placeholder for span_id=%s: %s",
            sid,
            exc,
        )
    try:
        if placeholder.metrics and trace_manager.is_evaluating:
            stash_pending_metrics(to_hex_string(sid, 16), placeholder.metrics)
    except Exception as exc:
        logger.debug(
            "Failed to stash pending metrics for span_id=%s: %s", sid, exc
        )


def serialize_trace_context_to_otel_attrs(
    span,
    instrumentation_settings,
    thread_id_fallback_attr: Optional[str] = None,
) -> None:
    """Resolve trace-level attrs FRESH and write them to ``confident.trace.*``.

    Reads ``current_trace_context`` (so ``update_current_trace(...)`` from
    anywhere in the call stack lands on every OTel span) with the
    instrumentation settings' trace defaults as fallback. Metadata merges
    settings as base + runtime context on top.

    Called at ``on_end`` (not ``on_start``) so the latest values are captured
    rather than a stale snapshot. Goes through ``set_span_attribute_post_end``
    because the SDK has already set ``_end_time`` by then, which makes
    ``span.set_attribute`` a silent no-op.

    ``thread_id_fallback_attr`` names a framework attribute (e.g. Strands'
    ``session.id``) to default the thread id from when nothing else set one.
    """
    trace_ctx = current_trace_context.get()
    conf = instrumentation_settings

    _name = (trace_ctx.name if trace_ctx else None) or conf.name
    _thread_id = (trace_ctx.thread_id if trace_ctx else None) or conf.thread_id
    _user_id = (trace_ctx.user_id if trace_ctx else None) or conf.user_id
    _tags = (trace_ctx.tags if trace_ctx else None) or conf.tags
    _test_case_id = (
        trace_ctx.test_case_id if trace_ctx else None
    ) or conf.test_case_id
    _turn_id = (trace_ctx.turn_id if trace_ctx else None) or conf.turn_id
    _trace_metric_collection = (
        trace_ctx.metric_collection if trace_ctx else None
    ) or conf.metric_collection
    _metadata = {
        **(conf.metadata or {}),
        **((trace_ctx.metadata or {}) if trace_ctx else {}),
    }

    if _name:
        set_span_attribute_post_end(span, ConfidentAttr.TRACE_NAME, _name)
    if _thread_id:
        set_span_attribute_post_end(
            span, ConfidentAttr.TRACE_THREAD_ID, _thread_id
        )
    if _user_id:
        set_span_attribute_post_end(span, ConfidentAttr.TRACE_USER_ID, _user_id)
    if _tags:
        set_span_attribute_post_end(span, ConfidentAttr.TRACE_TAGS, _tags)
    if _metadata:
        set_span_attribute_post_end(
            span, ConfidentAttr.TRACE_METADATA, serialize_to_json(_metadata)
        )
    if _trace_metric_collection:
        set_span_attribute_post_end(
            span,
            ConfidentAttr.TRACE_METRIC_COLLECTION,
            _trace_metric_collection,
        )
    if _test_case_id:
        set_span_attribute_post_end(
            span, ConfidentAttr.TRACE_TEST_CASE_ID, _test_case_id
        )
    if _turn_id:
        set_span_attribute_post_end(span, ConfidentAttr.TRACE_TURN_ID, _turn_id)
    if conf.environment:
        set_span_attribute_post_end(
            span, ConfidentAttr.TRACE_ENVIRONMENT, conf.environment
        )

    if thread_id_fallback_attr and not (span.attributes or {}).get(
        ConfidentAttr.TRACE_THREAD_ID
    ):
        session_id = (span.attributes or {}).get(thread_id_fallback_attr)
        if session_id:
            set_span_attribute_post_end(
                span, ConfidentAttr.TRACE_THREAD_ID, session_id
            )
