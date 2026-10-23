"""Run confident-trace as deepeval's tracing engine.

deepeval starts confident-trace itself. ``DeepEvalSpanBridge`` collects the
spans of deepeval-owned traces in-process, and confident-trace exports the rest
to Confident AI over OTLP under deepeval's API key.
"""

import logging
import threading
from collections import OrderedDict
from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.context import Context
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.sdk.trace import (
    ReadableSpan,
    Span,
    SpanProcessor,
    TracerProvider,
)
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.bridge import (
    OBSERVE_SCOPE,
    DeepEvalSpanBridge,
    RemoteSpanExporter,
    drop_from_export,
    trace_field_attributes,
)
from deepeval.tracing.otel.utils import otlp_traces_endpoint
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import BaseSpan
from deepeval.tracing.utils import make_json_serializable, tracing_enabled

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "confident-trace is not installed. Install it with "
    "`pip install confident-trace`."
)

# Above confident-trace's 16 KiB default so evaluated content such as
# retrieval context survives capture, while OTLP batches stay small.
_MAX_CONTENT_BYTES = 64 * 1024

_lock = threading.Lock()
_runtime: Any = None
_bridge: Optional[DeepEvalSpanBridge] = None
_tracer: Any = None
_instrumented: Set[str] = set()
# The attributes integrations start their spans with, by instrumentation scope
# or a prefix of its dotted name.
_scope_defaults: Dict[str, Dict[str, Any]] = {}


def ensure_confident_trace(
    instrumentations: Iterable[str] = (),
    *,
    api_key: Optional[str] = None,
    exporter: Optional[SpanExporter] = None,
    tracer_provider: Optional[TracerProvider] = None,
) -> bool:
    """Start confident-trace once, then enable any ``instrumentations`` not on.

    ``api_key``, when given, is the Confident AI key traces export under.
    ``exporter`` and ``tracer_provider`` replace the Confident AI exporter and
    the global provider, and only apply to the call that starts
    confident-trace. Returns whether confident-trace is running.
    """
    if api_key:
        trace_manager.configure(confident_api_key=api_key)
    try:
        import confident_trace
        from confident_trace.integrations import registry
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error

    global _runtime, _bridge
    with _lock:
        if _runtime is None or not _runtime.active:
            if _bridge is not None:
                _bridge.shutdown()
            _instrumented.clear()
            bridge = DeepEvalSpanBridge()
            runtime = confident_trace.init(
                exporter=RemoteSpanExporter(
                    exporter or ConfidentOtlpExporter()
                ),
                tracer_provider=tracer_provider,
                instrumentations=(),
                max_content_bytes=_MAX_CONTENT_BYTES,
            )
            if not runtime.active:
                return False
            runtime.provider.add_span_processor(_ScopeDefaults())
            runtime.provider.add_span_processor(bridge)
            global _tracer
            _runtime, _bridge = runtime, bridge
            _tracer = runtime.provider.get_tracer(OBSERVE_SCOPE)
        names = [
            name
            for name in dict.fromkeys(instrumentations)
            if name not in _instrumented
        ]
        if names:
            # ``init`` only instruments at start-up; the registry is how an
            # integration joins a confident-trace that is already running.
            _runtime.undo.extend(registry.instrument(_runtime, names))
            _instrumented.update(names)
        return True


def shutdown_confident_trace() -> None:
    """Stop confident-trace; ``ensure_confident_trace`` starts it again."""
    global _runtime, _bridge, _tracer
    with _lock:
        if _bridge is not None:
            _bridge.shutdown()
        if _runtime is not None:
            import confident_trace

            confident_trace.shutdown()
        _runtime = _bridge = _tracer = None
        _instrumented.clear()
        _scope_defaults.clear()


def set_trace_defaults(
    scope: str, values: Dict[str, Any], **attributes: Any
) -> None:
    """Start every span under instrumentation ``scope`` (or a scope it
    prefixes) with these trace fields and ``attributes``.

    ``update_current_trace(...)`` still overrides them.
    """
    _scope_defaults[scope] = {**trace_field_attributes(values), **attributes}


def flush_confident_trace(timeout: float) -> bool:
    """Wait up to ``timeout`` seconds for confident-trace to export its spans."""
    if _runtime is None or not _runtime.active:
        return True
    import confident_trace

    return bool(confident_trace.flush(int(timeout * 1000)))


def tracer_provider() -> Optional[TracerProvider]:
    """The provider confident-trace runs on, once deepeval has started it."""
    return (
        _runtime.provider if _runtime is not None and _runtime.active else None
    )


def current_collected_span() -> Optional[BaseSpan]:
    """The placeholder for the current confident-trace span, if collected."""
    if _bridge is None:
        return None
    return _bridge.span_for(otel_trace.get_current_span())


def start_observed_span(
    name: str, span_type: str
) -> Tuple[Optional[Span], Optional[BaseSpan]]:
    """Start an ``@observe`` span under the current one, not made current.

    Returns the span and, when deepeval collects it, its placeholder; no span
    when confident-trace isn't running.
    """
    if _tracer is None and not ensure_confident_trace():
        return None, None
    span = _tracer.start_span(
        name, attributes={ConfidentAttr.SPAN_TYPE: span_type}
    )
    return span, _bridge.span_for(span)


def observed_context(span: OtelSpan) -> Context:
    """The current context with ``span`` current, and the trace's entry when no
    span opened under it is yet.

    confident-trace writes a trace's own fields onto its entry, so marking it
    keeps a confident-trace span inside ``@observe`` from claiming the trace.
    """
    ctx = otel_trace.set_span_in_context(span)
    try:
        from confident_trace._core.spans import _ENTRY
    except ImportError:
        return ctx
    if otel_context.get_value(_ENTRY) is None:
        ctx = otel_context.set_value(_ENTRY, span, ctx)
    return ctx


def export_on(span: OtelSpan, scope: str = "span", **values: Any) -> None:
    """Export span or trace ``values`` on ``span``, if confident-trace exports it."""
    with otel_trace.use_span(span, end_on_exit=False):
        if scope == "trace":
            update_exported_trace(**values)
        else:
            update_exported_span(**values)


def exports(span: OtelSpan, attribute: str) -> bool:
    """Whether ``span`` already carries ``attribute``."""
    return attribute in (getattr(span, "attributes", None) or {})


def is_exported(span: Optional[OtelSpan]) -> bool:
    """Whether confident-trace will export the open ``span``."""
    return (
        span is not None and _bridge is not None and _bridge.is_exported(span)
    )


def open_collected_spans(trace_uuid: str) -> int:
    """How many collected spans of the deepeval trace are still open."""
    return _bridge.open_spans(trace_uuid) if _bridge is not None else 0


def drop_current(whole_trace: bool) -> None:
    """Keep the current span, or its whole trace, out of the export."""
    drop_from_export(otel_trace.get_current_span(), whole_trace)


def exported_trace_context(**values: Any) -> ContextManager:
    """Trace fields every span that starts a trace in this scope exports."""
    fields = _exportable(values, _EXPORTED_TRACE_FIELDS)
    if not fields or not ensure_confident_trace():
        return nullcontext()
    import confident_trace

    try:
        return confident_trace.trace_context(**fields)
    except (TypeError, ValueError):
        logger.debug("confident-trace rejected trace fields", exc_info=True)
        return nullcontext()


def update_exported_span(**values: Any) -> None:
    """Send span updates to the current span, if confident-trace exports it.

    Only fields the platform reads travel; metric objects stay local.
    """
    span = otel_trace.get_current_span()
    if is_exported(span):
        import confident_trace

        _send(confident_trace.update_span, values, _EXPORTED_SPAN_FIELDS)
        span.set_attributes(_platform_span_attributes(values))


def update_exported_trace(**values: Any) -> None:
    """Send trace updates to the current trace, if confident-trace exports it."""
    span = otel_trace.get_current_span()
    if not is_exported(span):
        return
    import confident_trace

    _send(confident_trace.update_trace, values, _EXPORTED_TRACE_FIELDS)
    if isinstance(values.get("test_run_id"), str):
        span.set_attribute(
            ConfidentAttr.TRACE_TEST_RUN_ID, values["test_run_id"]
        )
    if isinstance(values.get("confident_api_key"), str):
        _route_trace(span, values["confident_api_key"])


def suppress_confident_trace() -> ContextManager:
    """A scope that keeps deepeval's own model calls out of user traces.

    ``CONFIDENT_TRACE_INTERNAL`` opts back in, as it does for ``@observe``.
    """
    if (
        _runtime is None
        or not _runtime.active
        or get_settings().CONFIDENT_TRACE_INTERNAL
    ):
        return nullcontext()
    import confident_trace

    return confident_trace.suppress_tracing()


_CONTENT_FIELDS = (
    "input",
    "output",
    "metadata",
    "retrieval_context",
    "context",
    "expected_output",
    "tools_called",
    "expected_tools",
    "metric_collection",
)
_EXPORTED_SPAN_FIELDS = frozenset(
    (
        *_CONTENT_FIELDS,
        "name",
        "model",
        "provider",
        "input_token_count",
        "output_token_count",
        "cost_per_input_token",
        "cost_per_output_token",
    )
)
_EXPORTED_TRACE_FIELDS = frozenset(
    (
        *_CONTENT_FIELDS,
        "name",
        "tags",
        "thread_id",
        "user_id",
        "test_case_id",
        "turn_id",
    )
)
_TOOL_FIELDS = ("tools_called", "expected_tools")
_COUNT_FIELDS = ("input_token_count", "output_token_count")


# Span fields the platform reads under deepeval's own attributes, which
# confident-trace has none for.
_PLATFORM_SPAN_ATTRIBUTES = {
    "available_tools": ConfidentAttr.AGENT_AVAILABLE_TOOLS,
    "agent_handoffs": ConfidentAttr.AGENT_AGENT_HANDOFFS,
    "description": ConfidentAttr.TOOL_DESCRIPTION,
    "embedder": ConfidentAttr.RETRIEVER_EMBEDDER,
    "top_k": ConfidentAttr.RETRIEVER_TOP_K,
    "chunk_size": ConfidentAttr.RETRIEVER_CHUNK_SIZE,
    "integration": ConfidentAttr.SPAN_INTEGRATION,
}
_PROMPT_ATTRIBUTES = {
    "alias": ConfidentAttr.SPAN_PROMPT_ALIAS,
    "version": ConfidentAttr.SPAN_PROMPT_VERSION,
    "label": ConfidentAttr.SPAN_PROMPT_LABEL,
    "hash": ConfidentAttr.SPAN_PROMPT_COMMIT_HASH,
}


def _platform_span_attributes(values: Dict[str, Any]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    for key, attribute in _PLATFORM_SPAN_ATTRIBUTES.items():
        value = values.get(key)
        if isinstance(value, (list, tuple)):
            attributes[attribute] = [str(item) for item in value]
        elif isinstance(value, (str, int)) and not isinstance(value, bool):
            attributes[attribute] = value
    prompt = values.get("prompt")
    for key, attribute in _PROMPT_ATTRIBUTES.items():
        value = getattr(prompt, key, None) if prompt is not None else None
        if isinstance(value, str):
            attributes[attribute] = value
    return attributes


# The Confident AI key a trace exports under, when it isn't the default.
_MAX_ROUTED_TRACES = 10_000
_trace_keys: "OrderedDict[int, str]" = OrderedDict()
_trace_keys_lock = threading.Lock()


def _route_trace(span: OtelSpan, api_key: str) -> None:
    trace_id = span.get_span_context().trace_id
    with _trace_keys_lock:
        _trace_keys[trace_id] = api_key
        while len(_trace_keys) > _MAX_ROUTED_TRACES:
            _trace_keys.popitem(last=False)


def _trace_key(trace_id: int) -> Optional[str]:
    with _trace_keys_lock:
        return _trace_keys.get(trace_id)


def _exportable(
    values: Dict[str, Any], accepted: FrozenSet[str]
) -> Dict[str, Any]:
    """``values`` as confident-trace takes them, limited to ``accepted``."""
    values = dict(values)
    test_case = values.pop("test_case", None)
    if test_case is not None:
        values.update(
            input=test_case.input,
            output=test_case.actual_output,
            expected_output=test_case.expected_output,
            retrieval_context=test_case.retrieval_context,
            context=test_case.context,
            tools_called=test_case.tools_called,
            expected_tools=test_case.expected_tools,
        )
    fields = {}
    for key, value in values.items():
        if key not in accepted or value is None:
            continue
        if key in _TOOL_FIELDS:
            value = [call.model_dump(exclude_none=True) for call in value]
        elif key in _COUNT_FIELDS:
            value = int(value)
        elif key in _CONTENT_FIELDS:
            value = make_json_serializable(value)
        fields[key] = value
    return fields


def _send(
    update: Callable[..., None],
    values: Dict[str, Any],
    accepted: FrozenSet[str],
) -> None:
    fields = _exportable(values, accepted)
    if not fields:
        return
    try:
        update(**fields)
    except (TypeError, ValueError):
        # confident-trace rejects values it can't carry; user code goes on.
        logger.debug("confident-trace rejected an update", exc_info=True)


class _ScopeDefaults(SpanProcessor):
    """Stamps each span with its scope's defaults that it doesn't set."""

    def on_start(self, span: Span, parent_context=None) -> None:
        scope = span.instrumentation_scope
        if scope is None or not _scope_defaults:
            return
        present = span.attributes or {}
        for prefix, defaults in _scope_defaults.items():
            if scope.name == prefix or scope.name.startswith(f"{prefix}."):
                span.set_attributes(
                    {
                        key: value
                        for key, value in defaults.items()
                        if key not in present
                    }
                )


class ConfidentOtlpExporter(SpanExporter):
    """OTLP export to Confident AI under the API key resolved at export time.

    Resolving per export lets a key set after start-up take effect, and a
    trace may carry its own. Nothing is sent while tracing is off or for a
    trace with no key.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._exporters: Dict[str, SpanExporter] = {}

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not tracing_enabled() or not trace_manager.tracing_enabled:
            return SpanExportResult.SUCCESS
        default_key = trace_manager.confident_api_key or get_confident_api_key()
        by_key: Dict[str, List[ReadableSpan]] = {}
        for span in spans:
            api_key = _trace_key(span.context.trace_id) or default_key
            if api_key:
                by_key.setdefault(api_key, []).append(span)
        result = SpanExportResult.SUCCESS
        for api_key, keyed in by_key.items():
            if (
                self._exporter_for(api_key).export(keyed)
                != SpanExportResult.SUCCESS
            ):
                result = SpanExportResult.FAILURE
        return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            for exporter in self._exporters.values():
                exporter.shutdown()
            self._exporters.clear()

    def _exporter_for(self, api_key: str) -> SpanExporter:
        from opentelemetry.exporter.otlp.proto.http import Compression
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        with self._lock:
            if api_key not in self._exporters:
                self._exporters[api_key] = OTLPSpanExporter(
                    endpoint=otlp_traces_endpoint(),
                    headers={"x-confident-api-key": api_key},
                    compression=Compression.Gzip,
                )
            return self._exporters[api_key]
