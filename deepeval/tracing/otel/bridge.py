"""Collect the confident-trace spans that belong to deepeval-owned traces.

A span belongs to deepeval when it starts inside an active ``@observe`` or
``with trace(...)`` scope, during an evaluation, or while a trace-shape test is
capturing; its local descendants follow it, so a trace is never split. The
bridge converts those spans into ``trace_manager`` and marks them so
confident-trace's export skips them. Every other span is left to
confident-trace.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanContext

from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    current_trace_context,
    pop_pending_for,
)
from deepeval.tracing.otel.converter import (
    convert_span,
    is_ai_span,
    resolve_span_type,
)
from deepeval.tracing.otel.utils import (
    is_capturing_traces,
    set_trace_time,
    to_hex_string,
)
from deepeval.tracing.trace_context import (
    current_agent_context,
    current_llm_context,
)
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, SpanType, Trace, TraceSpanStatus

logger = logging.getLogger(__name__)

LOCAL_SPAN_ATTRIBUTE = "deepeval.local"

# Only a trace's root names it. Any span may fill an unset input or output,
# as deepeval's native integrations did.
_ROOT_TRACE_FIELDS = frozenset(("name",))

_SPAN_CONTEXTS = {
    SpanType.LLM: current_llm_context,
    SpanType.AGENT: current_agent_context,
}


@dataclass
class _Claim:
    """What a collected span took from the scope it started in."""

    trace_uuid: str
    parent_uuid: Optional[str]
    # deepeval-side values for the span, applied in order over what it recorded.
    defaults: List[Dict[str, Any]]


@dataclass
class _TraceState:
    # True when the bridge opened the deepeval trace and so must end it.
    owned: bool
    open_spans: int = 0
    spans: Dict[str, BaseSpan] = field(default_factory=dict)
    trace_fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # A dropped span's uuid, mapped to the parent its children attach to.
    lifted: Dict[str, Optional[str]] = field(default_factory=dict)


class DeepEvalSpanBridge(SpanProcessor):
    def __init__(self):
        self._lock = threading.RLock()
        # Every open span seen, so children inherit their parent's decision;
        # ``None`` marks a span left to confident-trace.
        self._claims: Dict[Tuple[int, int], Optional[_Claim]] = {}
        self._traces: Dict[str, _TraceState] = {}
        self._closed = False

    def on_start(self, span: Span, parent_context=None) -> None:
        if self._closed:
            return
        with self._lock:
            try:
                claim = self._claim(span)
            except Exception:
                logger.debug(
                    "Failed to claim span %s", span.name, exc_info=True
                )
                claim = None
            self._claims[_key(span.context)] = claim
        if claim is not None:
            span.set_attribute(LOCAL_SPAN_ATTRIBUTE, True)

    def on_end(self, span: ReadableSpan) -> None:
        with self._lock:
            claim = self._claims.pop(_key(span.context), None)
            if claim is None:
                return
            state = self._traces[claim.trace_uuid]
            try:
                self._collect(span, claim, state)
            except Exception:
                logger.debug(
                    "Failed to collect span %s", span.name, exc_info=True
                )
            state.open_spans -= 1
            if state.open_spans:
                return
            del self._traces[claim.trace_uuid]
            if not state.owned:
                return
            try:
                _end_owned_trace(claim.trace_uuid, state)
            except Exception:
                logger.debug(
                    "Failed to end trace %s", claim.trace_uuid, exc_info=True
                )

    def shutdown(self) -> None:
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _claim(self, span: Span) -> Optional[_Claim]:
        parent = span.parent
        if parent is not None and not parent.is_remote:
            parent_key = _key(parent)
            if parent_key in self._claims:
                parent_claim = self._claims[parent_key]
                if parent_claim is None:
                    return None
                return self._open(
                    span,
                    parent_claim.trace_uuid,
                    to_hex_string(parent.span_id, 16),
                )
        scope = _deepeval_scope()
        if scope is not None:
            return self._open(span, *scope, owned=False)
        if not is_capturing_traces():
            return None
        return self._open(
            span, to_hex_string(span.context.trace_id, 32), None, owned=True
        )

    def _open(
        self,
        span: Span,
        trace_uuid: str,
        parent_uuid: Optional[str],
        owned: bool = False,
    ) -> _Claim:
        span_type = resolve_span_type(span.attributes or {})
        defaults = []
        span_context = _SPAN_CONTEXTS.get(span_type)
        if span_context is not None and span_context.get() is not None:
            defaults.append(dict(vars(span_context.get())))
        # ``next_*_span`` staging is more specific, so it applies last.
        defaults.append(pop_pending_for(span_type.value if span_type else None))
        state = self._traces.get(trace_uuid)
        if state is None:
            if owned:
                # In an async evals_iterator this also binds the current golden.
                trace_manager.start_new_trace(trace_uuid=trace_uuid)
            state = self._traces[trace_uuid] = _TraceState(owned=owned)
        state.open_spans += 1
        return _Claim(trace_uuid, parent_uuid, defaults)

    def _collect(
        self, span: ReadableSpan, claim: _Claim, state: _TraceState
    ) -> None:
        trace = trace_manager.get_trace_by_uuid(claim.trace_uuid)
        if trace is None:
            # deepeval already ended the trace while this span was open.
            return
        uuid = to_hex_string(span.context.span_id, 16)
        parent_uuid = _lifted_parent(state, claim.parent_uuid)
        # As confident-trace exports them, non-AI spans stay only as ancestors.
        if not is_ai_span(span) and not any(
            child.parent_uuid == uuid for child in state.spans.values()
        ):
            state.lifted[uuid] = parent_uuid
            return
        converted = convert_span(span)
        collected = converted.span
        collected.trace_uuid = claim.trace_uuid
        collected.parent_uuid = parent_uuid
        for defaults in claim.defaults:
            apply_pending_to_span(collected, defaults)
        _attach(trace, state, collected)
        if state.owned:
            state.trace_fields[uuid] = converted.trace_fields
        else:
            _fill_unset(
                trace,
                converted.trace_fields,
                is_root=collected.parent_uuid is None,
            )


class RemoteSpanExporter(SpanExporter):
    """Exports only the spans the bridge left to confident-trace."""

    def __init__(self, exporter: SpanExporter):
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        remote = [
            span
            for span in spans
            if not (span.attributes or {}).get(LOCAL_SPAN_ATTRIBUTE)
        ]
        if not remote:
            return SpanExportResult.SUCCESS
        return self._exporter.export(remote)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._exporter.shutdown()


def _key(context: SpanContext) -> Tuple[int, int]:
    return context.trace_id, context.span_id


def _deepeval_scope() -> Optional[Tuple[str, Optional[str]]]:
    """The live deepeval trace, and span, that a span starting now joins."""
    span = current_span_context.get()
    if span is not None and trace_manager.get_trace_by_uuid(span.trace_uuid):
        return span.trace_uuid, span.uuid
    trace = current_trace_context.get()
    if (
        trace is not None
        and not trace._is_otel_implicit
        and trace_manager.get_trace_by_uuid(trace.uuid)
    ):
        return trace.uuid, None
    return None


def _lifted_parent(
    state: _TraceState, parent_uuid: Optional[str]
) -> Optional[str]:
    while parent_uuid in state.lifted:
        parent_uuid = state.lifted[parent_uuid]
    return parent_uuid


def _attach(trace: Trace, state: _TraceState, span: BaseSpan) -> None:
    # A parent that ended first (e.g. one returning a stream) is collected but
    # no longer active, so ``add_span_to_trace`` would not find it.
    parent = state.spans.get(span.parent_uuid)
    if parent is not None:
        parent.children.append(span)
        trace_manager._reparent_orphan_roots(trace, span)
    else:
        trace_manager.add_span_to_trace(span)
    state.spans[span.uuid] = span


def _fill_unset(
    trace: Trace, trace_fields: Dict[str, Any], is_root: bool
) -> None:
    """deepeval-owned traces keep their own values; spans only fill gaps."""
    for key, value in trace_fields.items():
        if key in _ROOT_TRACE_FIELDS and not is_root:
            continue
        if getattr(trace, key) is None:
            setattr(trace, key, value)


def _end_owned_trace(trace_uuid: str, state: _TraceState) -> None:
    trace = trace_manager.get_trace_by_uuid(trace_uuid)
    if trace is None:
        return
    if not state.spans:
        # Only non-AI spans ran, so there is nothing to evaluate or show.
        trace_manager.discard_trace(trace_uuid)
        return
    # Mirrors the platform: in depth-first order, later spans win.
    for span in _depth_first(trace.root_spans):
        for key, value in state.trace_fields.get(span.uuid, {}).items():
            setattr(trace, key, value)
    if any(root.status == TraceSpanStatus.ERRORED for root in trace.root_spans):
        trace.status = TraceSpanStatus.ERRORED
    set_trace_time(trace)
    trace_manager.end_trace(trace_uuid)


def _depth_first(spans: Sequence[BaseSpan]) -> Iterator[BaseSpan]:
    for span in sorted(spans, key=lambda span: span.uuid):
        yield span
        yield from _depth_first(span.children)
