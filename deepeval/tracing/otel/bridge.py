"""Collect the confident-trace spans that belong to deepeval-owned traces.

A span belongs to deepeval when it starts inside an active ``@observe`` or
``with trace(...)`` scope, during an evaluation, or while a trace-shape test is
capturing; its local descendants follow it, so a trace is never split. The
bridge converts those spans into ``trace_manager`` and marks them so
confident-trace's export skips them. Every other span is left to
confident-trace.

While a collected span is open, deepeval code sees it through a placeholder:
``update_current_span(...)`` writes to it and ``@observe`` nests under it.
Whatever is set on the placeholder overrides what the span recorded.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import SpanContext

from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    current_trace_context,
    is_descendant,
    pop_pending_for,
    pop_pending_span_type,
    pop_pending_trace,
)
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.test_case import ToolCall
from deepeval.tracing.otel.converter import (
    convert_span,
    is_ai_span,
    resolve_span_type,
)
from deepeval.tracing.otel.messages import latest_user_text
from deepeval.tracing.otel.utils import (
    is_capturing_traces,
    set_trace_time,
    to_hex_string,
)
from deepeval.tracing.utils import make_json_serializable
from deepeval.tracing.trace_context import (
    current_agent_context,
    current_llm_context,
)
from deepeval.tracing.tracing import (
    tool_call_of,
    tool_calls_under,
    trace_manager,
)
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    SpanType,
    ToolSpan,
    Trace,
    TraceSpanStatus,
)

logger = logging.getLogger(__name__)

LOCAL_SPAN_ATTRIBUTE = "deepeval.local"

# Only a trace's root names it. Any span may fill an unset input or output,
# as deepeval's native integrations did.
_ROOT_TRACE_FIELDS = frozenset(("name",))

_SPAN_CONTEXTS = {
    SpanType.LLM: current_llm_context,
    SpanType.AGENT: current_agent_context,
}

# The spans whose content describes the trace's own input and output.
_CONTENT_TYPES = (AgentSpan, LlmSpan)

_PLACEHOLDER_TYPES = {
    SpanType.LLM: LlmSpan,
    SpanType.AGENT: AgentSpan,
    SpanType.RETRIEVER: RetrieverSpan,
    SpanType.TOOL: ToolSpan,
}


@dataclass
class _Claim:
    """A collected span while it is open."""

    trace_uuid: str
    # What deepeval code sees and updates until the span ends; every field
    # set on it overrides what the span recorded.
    placeholder: BaseSpan
    # The type an integration staged for the span, over the one it records.
    span_type: Optional[SpanType] = None

    def overrides(self) -> Dict[str, Any]:
        return {
            key: getattr(self.placeholder, key)
            for key in self.placeholder.model_fields_set
        }


@dataclass
class _TraceState:
    # True when the bridge opened the deepeval trace and so must end it.
    owned: bool
    open_spans: int = 0
    spans: Dict[str, BaseSpan] = field(default_factory=dict)
    trace_fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    conversations: Dict[str, str] = field(default_factory=dict)
    # A dropped span's uuid, mapped to the parent its children attach to.
    lifted: Dict[str, Optional[str]] = field(default_factory=dict)
    # Trace fields the bridge derived from spans, with the span time that
    # chose each, so a better candidate may still replace them.
    derived: Dict[str, Tuple[float, Any]] = field(default_factory=dict)
    derived_tool_calls: Optional[List[ToolCall]] = None


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

    def span_for(self, span: OtelSpan) -> Optional[BaseSpan]:
        """The placeholder standing in for ``span`` while it is collected."""
        claim = self._claims.get(_key(span.get_span_context()))
        return claim.placeholder if claim is not None else None

    def is_exported(self, span: OtelSpan) -> bool:
        """Whether ``span`` is open and left to confident-trace's export."""
        key = _key(span.get_span_context())
        return key in self._claims and self._claims[key] is None

    def _claim(self, span: Span) -> Optional[_Claim]:
        parent = span.parent
        if parent is not None and not parent.is_remote:
            parent_key = _key(parent)
            if parent_key in self._claims:
                parent_claim = self._claims[parent_key]
                if parent_claim is None:
                    return None
                parent_uuid = parent_claim.placeholder.uuid
                # deepeval code opened inside the parent, such as an
                # ``@observe`` function, encloses this span.
                enclosing = current_span_context.get()
                if enclosing is not None and is_descendant(
                    enclosing, parent_uuid
                ):
                    parent_uuid = enclosing.uuid
                return self._open(span, parent_claim.trace_uuid, parent_uuid)
        # A root span: the trace it starts takes what an integration staged.
        staged = pop_pending_trace()
        scope = _deepeval_scope()
        if scope is not None:
            claim = self._open(span, *scope, owned=False)
        elif is_capturing_traces():
            claim = self._open(
                span, to_hex_string(span.context.trace_id, 32), None, owned=True
            )
        else:
            # On an exported root, ahead of the defaults confident-trace adds.
            span.set_attributes(trace_field_attributes(staged))
            return None
        trace = trace_manager.get_trace_by_uuid(claim.trace_uuid)
        for key, value in staged.items():
            if hasattr(trace, key):
                setattr(trace, key, value)
        return claim

    def _open(
        self,
        span: Span,
        trace_uuid: str,
        parent_uuid: Optional[str],
        owned: bool = False,
    ) -> _Claim:
        staged_type = pop_pending_span_type()
        span_type = (
            SpanType(staged_type)
            if staged_type
            else resolve_span_type(span, at_start=True)
        )
        placeholder = _PLACEHOLDER_TYPES.get(span_type, BaseSpan)(
            uuid=to_hex_string(span.context.span_id, 16),
            trace_uuid=trace_uuid,
            parent_uuid=parent_uuid,
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=peb.epoch_nanos_to_perf_seconds(span.start_time),
            name=span.name,
        )
        # Building set the identity fields; only later assignments override.
        placeholder.model_fields_set.clear()
        span_context = _SPAN_CONTEXTS.get(span_type)
        if span_context is not None and span_context.get() is not None:
            apply_pending_to_span(placeholder, vars(span_context.get()))
        # ``next_*_span`` staging is more specific, so it applies last.
        apply_pending_to_span(
            placeholder,
            pop_pending_for(span_type.value if span_type else None),
        )
        state = self._traces.get(trace_uuid)
        if state is None:
            if owned:
                # In an async evals_iterator this also binds the current golden.
                trace_manager.start_new_trace(trace_uuid=trace_uuid)
            state = self._traces[trace_uuid] = _TraceState(owned=owned)
        state.open_spans += 1
        return _Claim(
            trace_uuid, placeholder, span_type if staged_type else None
        )

    def _collect(
        self, span: ReadableSpan, claim: _Claim, state: _TraceState
    ) -> None:
        trace = trace_manager.get_trace_by_uuid(claim.trace_uuid)
        if trace is None:
            # deepeval already ended the trace while this span was open.
            return
        uuid = claim.placeholder.uuid
        parent_uuid = _lifted_parent(state, claim.placeholder.parent_uuid)
        # As confident-trace exports them, non-AI spans stay only as
        # ancestors. Anything that ended inside this span is an orphan root.
        if (
            claim.span_type is None
            and not is_ai_span(span)
            and not any(root.parent_uuid == uuid for root in trace.root_spans)
        ):
            state.lifted[uuid] = parent_uuid
            return
        converted = convert_span(span, claim.span_type)
        collected = converted.span
        collected.trace_uuid = claim.trace_uuid
        collected.parent_uuid = parent_uuid
        for key, value in claim.overrides().items():
            if hasattr(collected, key):
                setattr(collected, key, value)
        _attach(trace, state, collected)
        if isinstance(collected, AgentSpan):
            _fill_agent_content(collected)
        if not collected.tools_called:
            # Like ``@observe``, a span lists the tools run under it; an agent
            # also lists those its intermediate steps ran.
            collected.tools_called = (
                tool_calls_under(
                    collected, descendants=isinstance(collected, AgentSpan)
                )
                or collected.tools_called
            )
        if converted.conversation_id:
            state.conversations[uuid] = converted.conversation_id
        if state.owned:
            state.trace_fields[uuid] = converted.trace_fields
            return
        _fill_unset(
            trace,
            converted.trace_fields,
            is_root=collected.parent_uuid is None,
        )
        if trace.thread_id is None:
            trace.thread_id = converted.conversation_id
        _derive_content(trace, state, collected)
        if isinstance(collected, ToolSpan):
            _derive_tool_call(trace, state, collected)


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
    if trace is not None and trace_manager.get_trace_by_uuid(trace.uuid):
        return trace.uuid, None
    return None


_STAMPED_TRACE_STRINGS = {
    "name": ConfidentAttr.TRACE_NAME,
    "thread_id": ConfidentAttr.TRACE_THREAD_ID,
    "user_id": ConfidentAttr.TRACE_USER_ID,
    "test_case_id": ConfidentAttr.TRACE_TEST_CASE_ID,
    "turn_id": ConfidentAttr.TRACE_TURN_ID,
    "metric_collection": ConfidentAttr.TRACE_METRIC_COLLECTION,
    "environment": ConfidentAttr.TRACE_ENVIRONMENT,
}


def trace_field_attributes(values: Dict[str, Any]) -> Dict[str, Any]:
    """Trace fields as the ``confident.trace.*`` attributes that carry them."""
    attributes: Dict[str, Any] = {
        attribute: values[key]
        for key, attribute in _STAMPED_TRACE_STRINGS.items()
        if isinstance(values.get(key), str)
    }
    tags = values.get("tags")
    if isinstance(tags, (list, tuple)):
        attributes[ConfidentAttr.TRACE_TAGS] = [str(tag) for tag in tags]
    metadata = values.get("metadata")
    if isinstance(metadata, dict):
        attributes[ConfidentAttr.TRACE_METADATA] = json.dumps(
            make_json_serializable(metadata), separators=(",", ":")
        )
    return attributes


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
    if not trace.root_spans:
        # Only non-AI spans ran, so there is nothing to evaluate or show.
        trace_manager.discard_trace(trace_uuid)
        return
    # Mirrors the platform: in depth-first order, later spans win. Values from
    # ``update_current_trace(...)`` still win over all of them.
    spans = list(_depth_first(trace.root_spans))
    recorded: Dict[str, Any] = {}
    for span in spans:
        recorded.update(state.trace_fields.get(span.uuid, {}))
    _fill_unset(trace, recorded, is_root=True)
    _fill_unset(trace, _trace_content(trace, state, spans), is_root=True)
    if any(root.status == TraceSpanStatus.ERRORED for root in trace.root_spans):
        trace.status = TraceSpanStatus.ERRORED
    set_trace_time(trace)
    trace_manager.end_trace(trace_uuid)


def _trace_content(
    trace: Trace, state: _TraceState, spans: List[BaseSpan]
) -> Dict[str, Any]:
    """What the platform derives for a trace that its spans didn't fill.

    Agent and model spans give the name, the earliest input (narrowed to the
    latest user message) and the latest-ending output.
    """
    candidates = [span for span in spans if isinstance(span, _CONTENT_TYPES)]
    named = next(
        (span for span in candidates if isinstance(span, AgentSpan)),
        candidates[0] if candidates else None,
    )
    content_input, content_output = _content_of(candidates)
    roots = {
        state.conversations[root.uuid]
        for root in trace.root_spans
        if root.uuid in state.conversations
    }
    conversations = roots or set(state.conversations.values())
    return {
        "name": named.name if named else None,
        "input": content_input,
        "output": content_output,
        # Ambiguous when the spans name different conversations.
        "thread_id": (
            next(iter(conversations)) if len(conversations) == 1 else None
        ),
        "tools_called": [
            tool_call_of(span) for span in spans if isinstance(span, ToolSpan)
        ]
        or None,
    }


def _content_of(spans: Sequence[BaseSpan]) -> Tuple[Any, Any]:
    """The earliest input, narrowed to the latest user message, and the
    latest-ending output among ``spans``."""
    with_input = [span for span in spans if span.input is not None]
    with_output = [span for span in spans if span.output is not None]
    return (
        (
            latest_user_text(
                min(with_input, key=lambda span: span.start_time).input
            )
            if with_input
            else None
        ),
        (
            max(with_output, key=lambda span: span.end_time).output
            if with_output
            else None
        ),
    )


def _fill_agent_content(agent: AgentSpan) -> None:
    """An agent that records no content takes it from its own model calls
    and the agents it ran."""
    if agent.input is not None and agent.output is not None:
        return
    content_input, content_output = _content_of(list(_content_spans(agent)))
    if agent.input is None:
        agent.input = content_input
    if agent.output is None:
        agent.output = content_output


def _content_spans(span: BaseSpan) -> Iterator[BaseSpan]:
    """The model and agent spans under ``span`` that no nested agent owns."""
    for child in span.children:
        if isinstance(child, _CONTENT_TYPES):
            yield child
        if not isinstance(child, AgentSpan):
            yield from _content_spans(child)


def _derive_content(trace: Trace, state: _TraceState, span: BaseSpan) -> None:
    """``_trace_content``'s input and output, kept current as spans arrive."""
    if not isinstance(span, _CONTENT_TYPES):
        return
    if span.input is not None and _replaceable(
        trace, state, "input", lambda chosen: span.start_time < chosen
    ):
        _derive(
            trace, state, "input", span.start_time, latest_user_text(span.input)
        )
    if span.output is not None and _replaceable(
        trace, state, "output", lambda chosen: span.end_time > chosen
    ):
        _derive(trace, state, "output", span.end_time, span.output)


def _replaceable(
    trace: Trace, state: _TraceState, key: str, better: Callable[[float], bool]
) -> bool:
    """Unset, or derived from a worse span; never a value deepeval code set."""
    current = getattr(trace, key)
    if current is None:
        return True
    derived = state.derived.get(key)
    return derived is not None and current is derived[1] and better(derived[0])


def _derive(
    trace: Trace, state: _TraceState, key: str, chosen_at: float, value: Any
) -> None:
    setattr(trace, key, value)
    state.derived[key] = (chosen_at, value)


def _derive_tool_call(trace: Trace, state: _TraceState, span: ToolSpan) -> None:
    if trace.tools_called is None:
        state.derived_tool_calls = trace.tools_called = []
    if trace.tools_called is state.derived_tool_calls:
        trace.tools_called.append(tool_call_of(span))


def _depth_first(spans: Sequence[BaseSpan]) -> Iterator[BaseSpan]:
    for span in sorted(spans, key=lambda span: span.uuid):
        yield span
        yield from _depth_first(span.children)
