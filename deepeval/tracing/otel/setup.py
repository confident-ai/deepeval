"""Run confident-trace as deepeval's tracing engine.

deepeval starts confident-trace itself. ``DeepEvalSpanBridge`` collects the
spans of deepeval-owned traces in-process, and confident-trace exports the rest
to Confident AI over OTLP under deepeval's API key.
"""

import threading
from contextlib import nullcontext
from typing import Any, ContextManager, Iterable, Optional, Sequence, Set

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.tracing.otel.bridge import DeepEvalSpanBridge, RemoteSpanExporter
from deepeval.tracing.otel.utils import otlp_traces_endpoint
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.utils import tracing_enabled

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


def ensure_confident_trace(
    instrumentations: Iterable[str] = (),
    *,
    exporter: Optional[SpanExporter] = None,
    tracer_provider: Optional[TracerProvider] = None,
) -> bool:
    """Start confident-trace once, then enable any ``instrumentations`` not on.

    ``exporter`` and ``tracer_provider`` replace the Confident AI exporter and
    the global provider, and only apply to the call that starts
    confident-trace. Returns whether confident-trace is running.
    """
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
