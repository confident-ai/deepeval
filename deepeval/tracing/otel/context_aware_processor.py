"""Route spans using ownership captured synchronously at span start.

Explicit DeepEval scopes, golden-bound evaluations, and trace-shape tests enter
SpanCapture directly. Production spans alone enter the OTLP batch processor.
The exporter remains available for legacy explicit batch-export callers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from deepeval.config.settings import get_settings
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.tracing.tracing import trace_manager

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

    class _SpanProcessor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def on_start(self, span, parent_context):
            pass

        def on_end(self, span):
            pass

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis: int = 30_000):
            return True


if TYPE_CHECKING:
    from opentelemetry.sdk.trace import SpanProcessor


def _otlp_endpoint() -> str:
    settings = get_settings()
    return str(settings.CONFIDENT_OTEL_URL) + "v1/traces"


class ContextAwareSpanProcessor(_SpanProcessor):
    """Route OTel spans to REST or OTLP based on deepeval context state.

    Args:
        api_key: Optional Confident AI API key. When provided, used as
            the ``x-confident-api-key`` header for the OTLP exporter and
            forwarded to ``ConfidentSpanExporter`` for REST auth. When
            ``None``, both delegates are still wired up — local span
            translation continues to work — but outbound auth headers
            are omitted, so the Confident AI backend will reject the
            uploads. Pass a key when you actually want spans to land in
            Confident AI.
    """

    def __init__(self, api_key: Optional[str] = None):
        if not _OTEL_AVAILABLE:
            raise ImportError(
                "opentelemetry SDK is not installed. Install with "
                "`pip install opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-http`."
            )

        self._api_key = api_key
        self._retired_processors = []

        self._rest_exporter = ConfidentSpanExporter(api_key=api_key)
        self._rest_processor = SimpleSpanProcessor(self._rest_exporter)
        from deepeval.tracing.otel.capture import SpanCapture

        self._capture = SpanCapture(self._rest_exporter)
        # Only attach the auth header when we actually have a key — the
        # OTLPSpanExporter forwards the headers dict verbatim onto every
        # request, so a ``None`` value would either crash the gRPC/HTTP
        # client at send time or get serialized as the literal string
        # ``"None"`` server-side. Empty headers means the OTel pipeline
        # still runs (useful for local debugging) but the Confident AI
        # backend will reject the uploads.
        otlp_headers = {"x-confident-api-key": api_key} if api_key else {}
        self._otlp_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=_otlp_endpoint(),
                headers=otlp_headers,
            ),
        )

    def on_start(self, span, parent_context=None):
        self._capture.start(
            span, self._should_capture_locally(), api_key=self._api_key
        )
        binding = self._capture.bindings.get(self._capture.key(span))
        if binding is not None and binding.transport is None:
            binding.transport = self._otlp_processor.on_end

    @staticmethod
    def _should_capture_locally():
        # A process-wide iterator flag does not make unrelated tasks evaluable.
        from deepeval.contextvars import get_current_golden

        ctx = current_trace_context.get()
        return (
            (ctx is not None and not ctx._is_otel_implicit)
            or (
                trace_manager.is_evaluating
                and (
                    not trace_manager.is_iterator
                    or get_current_golden() is not None
                )
            )
            or trace_testing_manager.test_name is not None
        )

    def on_end(self, span):
        binding = self._capture.bindings.get(self._capture.key(span))
        transport = binding.transport if binding is not None else None
        if not self._capture.end(span):
            (transport or self._otlp_processor.on_end)(span)

    def reconfigure_api_key(self, api_key):
        if api_key == self._api_key:
            return
        self._retired_processors.append(self._otlp_processor)
        self._api_key = api_key
        self._otlp_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=_otlp_endpoint(),
                headers={"x-confident-api-key": api_key} if api_key else {},
            )
        )

    def shutdown(self):
        try:
            self._rest_processor.shutdown()
        except Exception as exc:
            logger.debug("REST processor shutdown failed: %s", exc)
        for retired in getattr(self, "_retired_processors", ()):
            retired.shutdown()
        try:
            self._otlp_processor.shutdown()
        except Exception as exc:
            logger.debug("OTLP processor shutdown failed: %s", exc)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Block until both transports have drained, or ``timeout_millis``.

        ``SimpleSpanProcessor.force_flush`` is a no-op that never reaches its
        exporter, so the REST exporter is drained explicitly. Without that,
        this reports success while traces are still queued on the trace
        worker thread.
        """
        ok_rest = True
        ok_otlp = True
        try:
            ok_processor = self._rest_processor.force_flush(timeout_millis)
            ok_exporter = self._rest_exporter.force_flush(timeout_millis)
            ok_rest = ok_processor and ok_exporter
        except Exception as exc:
            logger.debug("REST processor force_flush failed: %s", exc)
            ok_rest = False
        try:
            ok_otlp = self._otlp_processor.force_flush(timeout_millis)
        except Exception as exc:
            logger.debug("OTLP processor force_flush failed: %s", exc)
            ok_otlp = False
        for retired in getattr(self, "_retired_processors", ()):
            ok_otlp = retired.force_flush(timeout_millis) and ok_otlp
        return (
            ok_rest
            and ok_otlp
            and (not hasattr(self, "_capture") or self._capture.drained())
        )


__all__ = ["ContextAwareSpanProcessor"]
