"""Run confident-trace as deepeval's tracing engine.

deepeval starts confident-trace itself. ``DeepEvalSpanBridge`` collects the
spans of deepeval-owned traces in-process, and confident-trace exports the rest
to Confident AI over OTLP under deepeval's API key.
"""

import logging
import threading
from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    Iterable,
    Optional,
    Sequence,
    Set,
)

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import (
    ReadableSpan,
    Span,
    SpanProcessor,
    TracerProvider,
)
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.tracing.otel.bridge import (
    DeepEvalSpanBridge,
    RemoteSpanExporter,
    trace_field_attributes,
)
from deepeval.tracing.otel.utils import otlp_traces_endpoint
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import BaseSpan
from deepeval.tracing.utils import tracing_enabled

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "confident-trace is not installed. Install it with "
    '`pip install "deepeval[confident-trace]"` (Python 3.10+).'
)

# Above confident-trace's 16 KiB default so evaluated content such as
# retrieval context survives capture, while OTLP batches stay small.
_MAX_CONTENT_BYTES = 64 * 1024

_lock = threading.Lock()
_runtime: Any = None
_bridge: Optional[DeepEvalSpanBridge] = None
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
            _runtime, _bridge = runtime, bridge
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
    global _runtime, _bridge
    with _lock:
        if _bridge is not None:
            _bridge.shutdown()
        if _runtime is not None:
            import confident_trace

            confident_trace.shutdown()
        _runtime = _bridge = None
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


def update_exported_span(**values: Any) -> None:
    """Send span updates to the current span, if confident-trace exports it.

    Only fields confident-trace carries travel; metric objects stay local.
    """
    if _exported_span_is_current():
        import confident_trace

        _send(confident_trace.update_span, values, _EXPORTED_SPAN_FIELDS)


def update_exported_trace(**values: Any) -> None:
    """Send trace updates to the current trace, if confident-trace exports it."""
    if _exported_span_is_current():
        import confident_trace

        _send(confident_trace.update_trace, values, _EXPORTED_TRACE_FIELDS)


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


def _exported_span_is_current() -> bool:
    return _bridge is not None and _bridge.is_exported(
        otel_trace.get_current_span()
    )


def _send(
    update: Callable[..., None],
    values: Dict[str, Any],
    accepted: FrozenSet[str],
) -> None:
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
        fields[key] = value
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

    Resolving per export lets a key set after start-up take effect. As with
    trace posting, nothing is sent while tracing is off or no key is set.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._api_key: Optional[str] = None
        self._exporter: Optional[SpanExporter] = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not tracing_enabled() or not trace_manager.tracing_enabled:
            return SpanExportResult.SUCCESS
        api_key = trace_manager.confident_api_key or get_confident_api_key()
        if not api_key:
            return SpanExportResult.SUCCESS
        return self._exporter_for(api_key).export(spans)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            if self._exporter is not None:
                self._exporter.shutdown()

    def _exporter_for(self, api_key: str) -> SpanExporter:
        from opentelemetry.exporter.otlp.proto.http import Compression
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        with self._lock:
            if api_key != self._api_key:
                if self._exporter is not None:
                    self._exporter.shutdown()
                self._exporter = OTLPSpanExporter(
                    endpoint=otlp_traces_endpoint(),
                    headers={"x-confident-api-key": api_key},
                    compression=Compression.Gzip,
                )
                self._api_key = api_key
            return self._exporter
