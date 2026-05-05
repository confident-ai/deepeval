"""Context-aware OTel SpanProcessor used by deepeval's OTel integrations.

Routes each ended OTel span to one of two transports based on whether the
calling thread/task is inside a deepeval trace context (e.g. an ``@observe``
decorated function or a ``with trace(...)`` block) or an active evaluation
session:

  - REST path (``SimpleSpanProcessor(ConfidentSpanExporter())``) when
    ``current_trace_context`` is set OR ``trace_manager.is_evaluating`` is
    True OR trace-shape testing mode is active
    (``trace_testing_manager.test_name`` is set). This makes spans flow
    through ``trace_manager`` and unlocks pytest tracing evals +
    ``evals_iterator`` for OTel-based integrations, and lets the
    ``@assert_trace_json`` / ``@generate_trace_json`` test decorators
    capture trace-shape JSON for bare ``agent.run(...)`` callers (no
    ``@observe`` / ``with trace(...)`` wrapper) — the only path that
    populates ``trace_testing_manager.test_dict`` is
    ``trace_manager.end_trace``, which only fires on the REST path.

  - OTLP path (``BatchSpanProcessor(OTLPSpanExporter(...))``) otherwise.
    Direct push to Confident AI's OTel endpoint.

``on_start`` fires for both delegate processors (cheap; the SDK delegates
treat ``on_start`` as a no-op). ``on_end`` selects exactly one delegate so
spans are not double-exported. ``shutdown`` and ``force_flush`` forward to
both.
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

        self._rest_processor = SimpleSpanProcessor(
            ConfidentSpanExporter(api_key=api_key),
        )
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

    @staticmethod
    def _should_route_to_rest() -> bool:
        # User-pushed trace contexts (via ``@observe`` / ``with trace(...)``)
        # opt into REST routing through trace_manager. Implicit trace
        # placeholders pushed by an OTel SpanInterceptor (only present so
        # ``update_current_trace(...)`` works without an enclosing context)
        # do NOT count — those callers expect OTLP behavior.
        trace_ctx = current_trace_context.get()
        if trace_ctx is not None and not getattr(
            trace_ctx, "is_otel_implicit", False
        ):
            return True
        try:
            if trace_manager.is_evaluating:
                return True
        except Exception:
            pass
        # Trace-shape testing override: when a test harness has set
        # ``trace_testing_manager.test_name``, force REST so spans flow
        # through ``trace_manager.end_trace`` (the only writer of
        # ``trace_testing_manager.test_dict``). Otherwise the
        # ``@assert_trace_json`` decorator silently times out and compares
        # ``{}`` to ``{}``, which trivially passes — masking real
        # trace-shape regressions for bare ``agent.run(...)`` flows.
        try:
            return trace_testing_manager.test_name is not None
        except Exception:
            return False

    def on_start(self, span, parent_context=None):
        # Forward to both delegates. Both SDK-provided processors treat
        # on_start as a no-op, so this is cheap and side-effect-free.
        try:
            self._rest_processor.on_start(span, parent_context)
        except Exception as exc:
            logger.debug("REST processor on_start failed: %s", exc)
        try:
            self._otlp_processor.on_start(span, parent_context)
        except Exception as exc:
            logger.debug("OTLP processor on_start failed: %s", exc)

    def on_end(self, span):
        # Route to exactly one delegate to avoid double export.
        if self._should_route_to_rest():
            self._rest_processor.on_end(span)
        else:
            self._otlp_processor.on_end(span)

    def shutdown(self):
        try:
            self._rest_processor.shutdown()
        except Exception as exc:
            logger.debug("REST processor shutdown failed: %s", exc)
        try:
            self._otlp_processor.shutdown()
        except Exception as exc:
            logger.debug("OTLP processor shutdown failed: %s", exc)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        ok_rest = True
        ok_otlp = True
        try:
            ok_rest = self._rest_processor.force_flush(timeout_millis)
        except Exception as exc:
            logger.debug("REST processor force_flush failed: %s", exc)
            ok_rest = False
        try:
            ok_otlp = self._otlp_processor.force_flush(timeout_millis)
        except Exception as exc:
            logger.debug("OTLP processor force_flush failed: %s", exc)
            ok_otlp = False
        return ok_rest and ok_otlp


__all__ = ["ContextAwareSpanProcessor"]
