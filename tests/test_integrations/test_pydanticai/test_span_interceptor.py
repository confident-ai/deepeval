"""Unit tests for ``SpanInterceptor`` (Pydantic AI OTel integration).

Covers:
  - Trace-level reads from ``current_trace_context`` for ``thread_id``,
    ``name``, ``user_id``, ``tags``, ``metadata``, ``test_case_id``,
    ``turn_id``, and ``metric_collection`` — with
    ``DeepEvalInstrumentationSettings`` trace defaults as fallback when
    the runtime context doesn't set them.
  - Span-context push/pop: ``current_span_context`` is set to a
    placeholder ``BaseSpan`` for the OTel span's lifetime so
    ``update_current_span(...)`` works from anywhere in the call stack,
    and the placeholder's mutations are serialized back into
    ``confident.span.*`` OTel attributes at ``on_end``.
  - Implicit trace placeholder push for bare ``agent.run`` callers (so
    ``update_current_trace(...)`` works without ``@observe`` /
    ``with trace(...)``).
  - ``ContextAwareSpanProcessor`` routing logic (REST when a deepeval
    trace context is active or an evaluation is running, OTLP otherwise).
"""

import json
from itertools import count
from unittest.mock import MagicMock, patch

import pytest

from deepeval.integrations.pydantic_ai.instrumentator import SpanInterceptor
from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    current_trace_context,
    next_agent_span,
    next_llm_span,
    next_retriever_span,
    next_span,
    next_tool_span,
    pop_pending_for,
    update_current_span,
    update_current_trace,
)
from deepeval.tracing.types import AgentSpan, BaseSpan, TraceSpanStatus
from deepeval.tracing.otel.context_aware_processor import (
    ContextAwareSpanProcessor,
)
from deepeval.tracing.trace_context import trace


_span_id_counter = count(start=1)
_trace_id_counter = count(start=1)


def _make_mock_span(operation_name=None, agent_name=None, tool_name=None):
    """Mock OTel span that records ``set_attribute`` calls.

    Mirrors the real OTel SDK's invariant that ``Span.attributes`` is a view
    over the same underlying ``_attributes`` mapping — so writes via either
    ``set_attribute(...)`` or direct ``_attributes[...] = ...`` (used by
    ``SpanInterceptor._set_attr_post_end`` to bypass the ended-span guard)
    are observable via ``span.attributes.get(...)``.
    """
    span = MagicMock()
    backing: dict = {}
    span._attributes = backing
    span.attributes = backing
    if operation_name:
        backing["gen_ai.operation.name"] = operation_name
    if agent_name:
        backing["gen_ai.agent.name"] = agent_name
    if tool_name:
        backing["gen_ai.tool.name"] = tool_name
    span.set_attribute.side_effect = lambda k, v: backing.__setitem__(k, v)
    span.get_span_context.return_value = MagicMock(
        trace_id=next(_trace_id_counter),
        span_id=next(_span_id_counter),
    )
    span.parent = None
    span.start_time = None  # forces _push_span_context to use perf_counter()
    return span


def _make_settings(**kwargs):
    """Return a minimal mock ``DeepEvalInstrumentationSettings``.

    Only the attributes ``SpanInterceptor`` actually reads are populated.
    Anything not provided defaults to ``None`` so the
    context-vs-settings precedence logic is exercised cleanly.

    Settings now carries only trace-level fields (no per-span
    metric_collection / prompt / agent_metrics) — this mirrors the
    refactor that moved span-level configuration entirely to
    ``update_current_span(...)``. Trace-level ``metric_collection``
    remains because it lives on the ``Trace`` (not on a span).
    """
    settings = MagicMock(spec=[])  # spec=[] disallows auto-attrs
    settings.thread_id = kwargs.get("thread_id")
    settings.name = kwargs.get("name")
    settings.metadata = kwargs.get("metadata")
    settings.user_id = kwargs.get("user_id")
    settings.tags = kwargs.get("tags")
    settings.metric_collection = kwargs.get("metric_collection")
    settings.test_case_id = kwargs.get("test_case_id")
    settings.turn_id = kwargs.get("turn_id")
    settings.environment = kwargs.get("environment")
    return settings


# ---------------------------------------------------------------------------
# Trace-context reads (existing fields)
# ---------------------------------------------------------------------------


class TestSpanInterceptorTraceContextReads:
    def test_uses_settings_when_no_trace_context(self):
        """Falls back to settings when current_trace_context is None."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings(
                thread_id="settings-thread",
                name="settings-name",
                metadata={"source": "settings"},
            )
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)
            interceptor.on_end(span)

            assert (
                span.attributes.get("confident.trace.thread_id")
                == "settings-thread"
            )
            assert (
                span.attributes.get("confident.trace.name") == "settings-name"
            )
            assert json.loads(span.attributes["confident.trace.metadata"]) == {
                "source": "settings"
            }
        finally:
            current_trace_context.reset(token)

    def test_prefers_trace_context_over_settings_for_scalars(self):
        """thread_id and name from current_trace_context override settings."""
        settings = _make_settings(
            thread_id="settings-thread",
            name="settings-name",
            metadata={"settings_key": "settings_val"},
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(
            thread_id="ctx-thread",
            name="ctx-name",
            metadata={"ctx_key": "ctx_val"},
        ):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.thread_id") == "ctx-thread"
        assert span.attributes.get("confident.trace.name") == "ctx-name"

    def test_metadata_is_merged_with_context_winning(self):
        """metadata from settings + current_trace_context merge; context wins."""
        settings = _make_settings(
            metadata={"base_key": "base_val", "shared_key": "from_settings"},
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(metadata={"ctx_key": "ctx_val", "shared_key": "from_ctx"}):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        result = json.loads(span.attributes["confident.trace.metadata"])
        assert result["base_key"] == "base_val"
        assert result["ctx_key"] == "ctx_val"
        assert result["shared_key"] == "from_ctx"

    def test_no_attributes_set_when_all_none(self):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)
            interceptor.on_end(span)

            assert "confident.trace.thread_id" not in span.attributes
            assert "confident.trace.name" not in span.attributes
            assert "confident.trace.metadata" not in span.attributes
            assert "confident.trace.user_id" not in span.attributes
            assert "confident.trace.tags" not in span.attributes
        finally:
            current_trace_context.reset(token)


# ---------------------------------------------------------------------------
# Trace-context reads (new in Phase 2)
# ---------------------------------------------------------------------------


class TestSpanInterceptorNewTraceContextReads:
    def test_user_id_from_trace_context_overrides_settings(self):
        settings = _make_settings(user_id="settings-user")
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(user_id="ctx-user"):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.user_id") == "ctx-user"

    def test_tags_from_trace_context_overrides_settings(self):
        settings = _make_settings(tags=["settings-tag"])
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(tags=["ctx-tag-1", "ctx-tag-2"]):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert list(span.attributes.get("confident.trace.tags")) == [
            "ctx-tag-1",
            "ctx-tag-2",
        ]

    def test_test_case_id_and_turn_id_from_trace_context_override_settings(
        self,
    ):
        settings = _make_settings(
            test_case_id="settings-tc",
            turn_id="settings-turn",
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace():
            update_current_trace(test_case_id="ctx-tc", turn_id="ctx-turn")
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.test_case_id") == "ctx-tc"
        assert span.attributes.get("confident.trace.turn_id") == "ctx-turn"

    def test_trace_metric_collection_resolution_order(self):
        """``metric_collection`` resolves runtime-context-first, settings
        as fallback — same precedence as the other trace-level fields.
        The runtime call wins on the value it touches."""
        settings = _make_settings(metric_collection="settings-mc")
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(metric_collection="ctx-mc"):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert (
            span.attributes.get("confident.trace.metric_collection") == "ctx-mc"
        )

    def test_update_current_trace_after_on_start_lands_on_otel_attrs(self):
        """Trace attrs are snapshotted FRESH at on_end, not on_start.

        Regression guard for the trace-attrs-at-on_start asymmetry: if a
        downstream caller mutates the active trace via ``update_current_trace``
        AFTER the OTel span's ``on_start`` has fired (e.g. from inside an
        ``@agent.tool_plain`` body or any nested helper), the new values
        must still land on this span's ``confident.trace.*`` OTel attributes
        when ``on_end`` runs.
        """
        settings = _make_settings(name="settings-name", user_id="settings-user")
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(name="initial-name"):
            interceptor.on_start(span, None)

            update_current_trace(
                name="updated-name",
                user_id="updated-user",
                tags=["after-update"],
                metadata={"phase": "post-start"},
            )

            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.name") == "updated-name"
        assert span.attributes.get("confident.trace.user_id") == "updated-user"
        assert list(span.attributes.get("confident.trace.tags")) == [
            "after-update"
        ]
        assert json.loads(span.attributes["confident.trace.metadata"]) == {
            "phase": "post-start"
        }

    def test_trace_metric_collection_falls_back_to_settings(self):
        """Without a runtime ``metric_collection`` set, the
        ``DeepEvalInstrumentationSettings`` default is used — same
        fallback behavior as ``name`` / ``user_id`` / etc."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings(metric_collection="settings-mc")
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)
            interceptor.on_end(span)

            assert (
                span.attributes.get("confident.trace.metric_collection")
                == "settings-mc"
            )
        finally:
            current_trace_context.reset(token)

    def test_trace_metric_collection_omitted_when_neither_set(self):
        """No ``confident.trace.metric_collection`` attr is written when
        neither settings nor the runtime context provide a value."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)
            interceptor.on_end(span)

            assert (
                span.attributes.get("confident.trace.metric_collection") is None
            )
        finally:
            current_trace_context.reset(token)


# ---------------------------------------------------------------------------
# Span-context push/pop: enables update_current_span(...) from anywhere
# ---------------------------------------------------------------------------


class TestSpanInterceptorSpanContextPushPop:
    def test_current_span_context_set_during_span_lifetime(self):
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        # Outside the span, current_span_context.get() may be None or a stale
        # sentinel; we only assert about the *change* introduced by on_start.
        before = current_span_context.get()
        interceptor.on_start(span, None)
        during = current_span_context.get()

        assert during is not None
        assert during is not before

        interceptor.on_end(span)
        after = current_span_context.get()
        assert after is before

    def test_update_current_span_metadata_lands_in_otel_attrs(self):
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        interceptor.on_start(span, None)
        update_current_span(
            metadata={"weather_source": "mock", "city": "Paris"},
            input={"query": "Weather?"},
            output="Sunny",
        )
        interceptor.on_end(span)

        assert span.attributes.get("confident.span.metadata") is not None
        assert json.loads(span.attributes["confident.span.metadata"]) == {
            "weather_source": "mock",
            "city": "Paris",
        }
        assert json.loads(span.attributes["confident.span.input"]) == {
            "query": "Weather?"
        }
        assert json.loads(span.attributes["confident.span.output"]) == "Sunny"

    def test_update_current_span_metric_collection_lands_in_otel_attrs(self):
        """update_current_span(metric_collection=...) overwrites placeholder."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        interceptor.on_start(span, None)
        update_current_span(metric_collection="runtime-collection")
        interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "runtime-collection"
        )

    def test_nested_spans_lifo_pop_restores_parent_placeholder(self):
        """Inner span's on_end restores the outer span's placeholder."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        outer = _make_mock_span()
        inner = _make_mock_span()

        interceptor.on_start(outer, None)
        outer_placeholder = current_span_context.get()

        interceptor.on_start(inner, None)
        inner_placeholder = current_span_context.get()
        assert inner_placeholder is not outer_placeholder

        interceptor.on_end(inner)
        assert current_span_context.get() is outer_placeholder

        interceptor.on_end(outer)


# ---------------------------------------------------------------------------
# Implicit trace context push/pop: enables update_current_trace(...) without
# an enclosing @observe / with trace(...) (bare ``agent.run`` callers).
# ---------------------------------------------------------------------------


class TestSpanInterceptorImplicitTraceContext:
    """Symmetric to ``TestSpanInterceptorSpanContextPushPop`` but at the
    trace level. The interceptor pushes an implicit ``Trace`` placeholder
    onto ``current_trace_context`` for the OTel root span's lifetime so
    ``update_current_trace(...)`` from inside tools / nested helpers can
    mutate something. The placeholder is tagged ``is_otel_implicit=True``
    so ``ContextAwareSpanProcessor`` keeps routing to OTLP.
    """

    def test_root_span_pushes_implicit_trace_when_no_user_context(self):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            root = _make_mock_span()  # parent=None by default

            interceptor.on_start(root, None)
            during = current_trace_context.get()

            assert during is not None
            assert getattr(during, "is_otel_implicit", False) is True

            interceptor.on_end(root)
            assert current_trace_context.get() is None
        finally:
            current_trace_context.reset(token)

    def test_does_not_overwrite_user_pushed_trace_context(self):
        """If the caller is already inside @observe / with trace(...),
        the interceptor must NOT clobber their Trace."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        root = _make_mock_span()

        with trace() as user_trace:
            assert getattr(user_trace, "is_otel_implicit", False) is False

            interceptor.on_start(root, None)
            during = current_trace_context.get()

            # Same object as the user's trace — no implicit push happened.
            assert during is user_trace
            assert getattr(during, "is_otel_implicit", False) is False

            interceptor.on_end(root)

            # User trace still in place after on_end (nothing was popped
            # because nothing was pushed).
            assert current_trace_context.get() is user_trace

    def test_child_span_does_not_push_its_own_placeholder(self):
        """Only the OTel root span pushes; child spans inherit via
        contextvars and never call ``current_trace_context.set``.
        """
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            root = _make_mock_span()
            child = _make_mock_span()
            child.parent = MagicMock()  # non-None marks it as a child

            interceptor.on_start(root, None)
            implicit = current_trace_context.get()
            assert implicit is not None

            interceptor.on_start(child, None)
            # Child sees the same implicit placeholder via contextvars; no
            # second push happened.
            assert current_trace_context.get() is implicit

            interceptor.on_end(child)
            # Child's on_end must not pop the root's placeholder.
            assert current_trace_context.get() is implicit

            interceptor.on_end(root)
            assert current_trace_context.get() is None
        finally:
            current_trace_context.reset(token)

    def test_update_current_trace_in_implicit_context_lands_on_otel_attrs(
        self,
    ):
        """The whole point of the implicit push: bare callers can use
        ``update_current_trace(...)`` from inside a tool body and have
        the values flow into ``confident.trace.*`` OTel attrs.
        """
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            root = _make_mock_span()

            interceptor.on_start(root, None)
            update_current_trace(
                name="bare-trace",
                user_id="user-bare",
                tags=["bare"],
                metadata={"source": "tool", "request_id": "req-bare-1"},
            )
            interceptor.on_end(root)

            assert root.attributes.get("confident.trace.name") == "bare-trace"
            assert root.attributes.get("confident.trace.user_id") == "user-bare"
            assert root.attributes.get("confident.trace.tags") == ["bare"]
            assert json.loads(root.attributes["confident.trace.metadata"]) == {
                "source": "tool",
                "request_id": "req-bare-1",
            }
        finally:
            current_trace_context.reset(token)


# ---------------------------------------------------------------------------
# ContextAwareSpanProcessor routing
# ---------------------------------------------------------------------------


class _FakeSpan:
    """Minimal stand-in for an OTel span with a stable identity."""


class TestContextAwareSpanProcessorRouting:
    @staticmethod
    def _make_processor():
        """Bypass ``__init__`` so the test doesn't depend on the OTLP exporter
        package being installed locally — we only care about routing logic.
        """
        processor = ContextAwareSpanProcessor.__new__(ContextAwareSpanProcessor)
        processor._api_key = "test-key"
        processor._rest_processor = MagicMock()
        processor._otlp_processor = MagicMock()
        return processor, processor._rest_processor, processor._otlp_processor

    def test_routes_to_rest_when_trace_context_active(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        with trace():
            processor.on_end(span)

        rest.on_end.assert_called_once_with(span)
        otlp.on_end.assert_not_called()

    def test_routes_to_otlp_when_no_context(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        token = current_trace_context.set(None)
        try:
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = False
                processor.on_end(span)
        finally:
            current_trace_context.reset(token)

        otlp.on_end.assert_called_once_with(span)
        rest.on_end.assert_not_called()

    def test_routes_to_rest_when_evaluating(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        token = current_trace_context.set(None)
        try:
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = True
                processor.on_end(span)
        finally:
            current_trace_context.reset(token)

        rest.on_end.assert_called_once_with(span)
        otlp.on_end.assert_not_called()

    def test_routes_to_otlp_when_only_implicit_trace_in_context(self):
        """Implicit Trace placeholders (pushed by SpanInterceptor for
        bare ``agent.run`` callers) MUST NOT flip routing to REST —
        they only exist so ``update_current_trace(...)`` works."""
        from deepeval.tracing.types import Trace, TraceSpanStatus

        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        implicit_trace = Trace(
            uuid="abc",
            root_spans=[],
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=0.0,
            is_otel_implicit=True,
        )
        token = current_trace_context.set(implicit_trace)
        try:
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = False
                processor.on_end(span)
        finally:
            current_trace_context.reset(token)

        otlp.on_end.assert_called_once_with(span)
        rest.on_end.assert_not_called()

    def test_routes_to_rest_when_evaluating_even_with_implicit_trace(self):
        """``trace_manager.is_evaluating`` overrides everything — a live
        eval session must see spans via REST regardless of how the trace
        context was pushed."""
        from deepeval.tracing.types import Trace, TraceSpanStatus

        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        implicit_trace = Trace(
            uuid="abc",
            root_spans=[],
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=0.0,
            is_otel_implicit=True,
        )
        token = current_trace_context.set(implicit_trace)
        try:
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = True
                processor.on_end(span)
        finally:
            current_trace_context.reset(token)

        rest.on_end.assert_called_once_with(span)
        otlp.on_end.assert_not_called()

    def test_routes_to_rest_when_test_name_is_set(self):
        """Trace-shape testing override: when
        ``trace_testing_manager.test_name`` is set (i.e. inside an
        ``@assert_trace_json`` / ``@generate_trace_json`` decorator),
        spans must flow through the REST path even with no user-pushed
        trace context and ``is_evaluating=False`` — otherwise the only
        writer of ``trace_testing_manager.test_dict``
        (``trace_manager.end_trace``) never fires for bare
        ``agent.run(...)`` flows, the decorator's
        ``wait_for_test_dict()`` times out, and ``{} == {}`` makes
        every schema test trivially pass.
        """
        from deepeval.tracing.trace_test_manager import (
            trace_testing_manager,
        )

        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        token = current_trace_context.set(None)
        prev_test_name = trace_testing_manager.test_name
        try:
            trace_testing_manager.test_name = "any_name"
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = False
                processor.on_end(span)
        finally:
            trace_testing_manager.test_name = prev_test_name
            current_trace_context.reset(token)

        rest.on_end.assert_called_once_with(span)
        otlp.on_end.assert_not_called()

    def test_routes_to_rest_when_test_name_set_with_implicit_trace(self):
        """The actual scenario in ``test_sync.py`` / ``test_async.py``:
        bare ``agent.run(...)`` (so ``SpanInterceptor`` pushes an implicit
        ``Trace`` placeholder, which on its own routes to OTLP) PLUS the
        test harness has set ``trace_testing_manager.test_name``. The
        test-name override must still flip routing to REST even though
        the only trace context active is implicit.
        """
        from deepeval.tracing.trace_test_manager import (
            trace_testing_manager,
        )
        from deepeval.tracing.types import Trace, TraceSpanStatus

        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        implicit_trace = Trace(
            uuid="abc",
            root_spans=[],
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=0.0,
            is_otel_implicit=True,
        )
        token = current_trace_context.set(implicit_trace)
        prev_test_name = trace_testing_manager.test_name
        try:
            trace_testing_manager.test_name = "any_name"
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = False
                processor.on_end(span)
        finally:
            trace_testing_manager.test_name = prev_test_name
            current_trace_context.reset(token)

        rest.on_end.assert_called_once_with(span)
        otlp.on_end.assert_not_called()

    def test_routes_to_otlp_when_test_name_is_none(self):
        """Negative guard: a freshly-cleared ``test_name`` (the default
        outside the test decorators) must NOT spuriously route to REST.
        Pairs with ``test_routes_to_rest_when_test_name_is_set`` so a
        future bug that flips the predicate (e.g. ``is None`` vs
        ``is not None``) is caught immediately.
        """
        from deepeval.tracing.trace_test_manager import (
            trace_testing_manager,
        )

        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        token = current_trace_context.set(None)
        prev_test_name = trace_testing_manager.test_name
        try:
            trace_testing_manager.test_name = None
            with patch(
                "deepeval.tracing.otel.context_aware_processor.trace_manager"
            ) as fake_tm:
                fake_tm.is_evaluating = False
                processor.on_end(span)
        finally:
            trace_testing_manager.test_name = prev_test_name
            current_trace_context.reset(token)

        otlp.on_end.assert_called_once_with(span)
        rest.on_end.assert_not_called()

    def test_on_start_forwarded_to_both(self):
        processor, rest, otlp = self._make_processor()
        span = _FakeSpan()

        processor.on_start(span, None)

        rest.on_start.assert_called_once_with(span, None)
        otlp.on_start.assert_called_once_with(span, None)

    def test_shutdown_and_force_flush_forwarded_to_both(self):
        processor, rest, otlp = self._make_processor()

        rest.force_flush.return_value = True
        otlp.force_flush.return_value = True

        assert processor.force_flush(timeout_millis=5000) is True
        rest.force_flush.assert_called_once_with(5000)
        otlp.force_flush.assert_called_once_with(5000)

        processor.shutdown()
        rest.shutdown.assert_called_once_with()
        otlp.shutdown.assert_called_once_with()


# ---------------------------------------------------------------------------
# Pytest signal: is_test_mode is gone for good.
# ---------------------------------------------------------------------------


def test_is_test_mode_kwarg_is_removed_from_settings():
    """Phase 2 hard-removed the kwarg. Calling with it must raise TypeError."""
    from deepeval.integrations.pydantic_ai.instrumentator import (
        DeepEvalInstrumentationSettings,
    )

    with pytest.raises(TypeError):
        DeepEvalInstrumentationSettings(api_key="dummy", is_test_mode=False)


# ---------------------------------------------------------------------------
# Span-related kwargs are gone for good — they intentionally have NO
# settings-level fallback. Per-span configuration is a runtime concern
# (``update_current_span(...)`` from inside your tool / agent body).
#
# ``metric_collection`` is NOT in this list — it lives on the ``Trace``
# (a trace-level field, alongside ``name`` / ``tags`` / etc.) and remains
# an accepted ``DeepEvalInstrumentationSettings`` kwarg as a
# trace-default. ``trace_metric_collection`` was a redundant alias and IS
# removed; use ``metric_collection`` instead.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwarg",
    [
        "confident_prompt",
        "trace_metric_collection",
        "llm_metric_collection",
        "agent_metric_collection",
        "tool_metric_collection_map",
        "agent_metrics",
    ],
)
def test_span_related_kwargs_are_removed_from_settings(kwarg):
    """Dropped span-level kwargs must raise TypeError on construction."""
    from deepeval.integrations.pydantic_ai.instrumentator import (
        DeepEvalInstrumentationSettings,
    )

    with pytest.raises(TypeError):
        DeepEvalInstrumentationSettings(api_key="dummy", **{kwarg: object()})


# ---------------------------------------------------------------------------
# Optional Confident AI api_key — the integration must NOT require a key.
#
# Historical behavior was a hard ``raise ValueError("CONFIDENT_API_KEY is
# not set")`` from the constructor when neither an explicit ``api_key``
# nor the ``CONFIDENT_API_KEY`` env var was present. That coupled the
# whole pydantic-ai OTel pipeline to having a Confident AI account. The
# rename to ``DeepEvalInstrumentationSettings`` lifts that requirement —
# without a key the OTel pipeline still wires up locally; only the
# outbound auth header is omitted.
# ---------------------------------------------------------------------------


def test_no_api_key_does_not_raise(monkeypatch):
    """Constructor must succeed when no api_key is supplied or in env."""
    from deepeval.integrations.pydantic_ai.instrumentator import (
        DeepEvalInstrumentationSettings,
    )

    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)
    monkeypatch.setattr(
        "deepeval.integrations.pydantic_ai.instrumentator."
        "get_confident_api_key",
        lambda: None,
    )

    instance = DeepEvalInstrumentationSettings()
    assert instance is not None


# ---------------------------------------------------------------------------
# Backward compatibility: ``ConfidentInstrumentationSettings`` deprecated
# alias must still construct (with a DeprecationWarning) and behave like
# the new class.
# ---------------------------------------------------------------------------


def test_confident_alias_emits_deprecation_warning(monkeypatch):
    """The old name still works but warns at instantiation time."""
    from deepeval.integrations.pydantic_ai.instrumentator import (
        ConfidentInstrumentationSettings,
        DeepEvalInstrumentationSettings,
    )

    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)
    monkeypatch.setattr(
        "deepeval.integrations.pydantic_ai.instrumentator."
        "get_confident_api_key",
        lambda: None,
    )

    with pytest.warns(DeprecationWarning, match="ConfidentInstrumentation"):
        instance = ConfidentInstrumentationSettings()

    # Subclass relationship — anywhere typed against the new name must
    # still accept old-name instances.
    assert isinstance(instance, DeepEvalInstrumentationSettings)


# ---------------------------------------------------------------------------
# next_*_span context managers — pure context-API behavior.
#
# These tests don't touch the SpanInterceptor; they verify the
# ``pop_pending_for(...)`` / ``apply_pending_to_span(...)`` contracts in
# isolation so we can assert the consumption semantics independently of
# any integration that wires them up.
# ---------------------------------------------------------------------------


class TestNextSpanPureContextAPI:
    def test_pop_outside_with_returns_empty(self):
        """No pending slot → popping returns an empty dict, never None."""
        assert pop_pending_for("agent") == {}
        assert pop_pending_for(None) == {}

    def test_next_agent_span_one_shot_consumption(self):
        """First pop drains; second pop returns empty for the same scope."""
        with next_agent_span(metric_collection="A", available_tools=["x"]):
            first = pop_pending_for("agent")
            assert first == {
                "metric_collection": "A",
                "available_tools": ["x"],
            }

            second = pop_pending_for("agent")
            assert second == {}

    def test_scope_exit_restores_prior_value(self):
        """Token-based reset: leaving the ``with`` block puts the slot
        back to ``None`` (not just empty-dict)."""
        with next_agent_span(metric_collection="A"):
            pass

        # After exit, popping yields nothing — slot is back to None.
        assert pop_pending_for("agent") == {}

    def test_stacked_typed_slots_are_independent(self):
        """``with next_agent_span(...), next_llm_span(...):`` keeps each
        slot separate; popping one does not drain the other."""
        with next_agent_span(metric_collection="A"), next_llm_span(
            model="gpt-4"
        ):
            agent_payload = pop_pending_for("agent")
            assert agent_payload == {"metric_collection": "A"}

            llm_payload = pop_pending_for("llm")
            assert llm_payload == {"model": "gpt-4"}

    def test_base_slot_consumed_by_first_typed_pop(self):
        """``next_span`` is "next of any type"; the first ``pop_pending_for``
        call inside the scope drains it regardless of typed slot match."""
        with next_span(metadata={"k": "v"}):
            first = pop_pending_for("agent")
            assert first == {"metadata": {"k": "v"}}

            # Subsequent pops see no base slot.
            assert pop_pending_for("llm") == {}

    def test_typed_overrides_base_on_key_overlap(self):
        """When base + typed both set the same key, the typed slot wins
        (more specific wins)."""
        with next_span(metric_collection="base"), next_agent_span(
            metric_collection="typed"
        ):
            payload = pop_pending_for("agent")
            assert payload["metric_collection"] == "typed"

    def test_pop_for_mismatched_type_drains_only_base(self):
        """``next_agent_span(...)`` is NOT consumed by
        ``pop_pending_for('llm')``. Base slot still goes (it's
        any-type)."""
        with next_span(metadata={"k": "v"}), next_agent_span(
            metric_collection="A"
        ):
            llm_payload = pop_pending_for("llm")
            # Base flowed through, agent slot untouched.
            assert llm_payload == {"metadata": {"k": "v"}}

            agent_payload = pop_pending_for("agent")
            assert agent_payload == {"metric_collection": "A"}

    def test_nested_same_type_innermost_wins(self):
        """Nested ``with next_agent_span(...)`` blocks: inner overrides
        for its scope; on exit, outer is restored."""
        with next_agent_span(metric_collection="outer"):
            with next_agent_span(metric_collection="inner"):
                assert pop_pending_for("agent") == {
                    "metric_collection": "inner"
                }

            # Outer scope's value is back, ready to be consumed once.
            assert pop_pending_for("agent") == {"metric_collection": "outer"}

    def test_drops_none_kwargs(self):
        """Slots store only kwargs the user actually passed; ``None``
        kwargs are stripped so consumers don't have to re-check."""
        with next_agent_span(metric_collection="A"):
            payload = pop_pending_for("agent")
            assert "available_tools" not in payload
            assert "name" not in payload
            assert "metadata" not in payload

    def test_unconsumed_payload_does_not_leak_across_scopes(self):
        """If no consumer pops inside the ``with``, the payload is
        discarded on exit — it never leaks to a sibling scope."""
        with next_agent_span(metric_collection="leaked"):
            pass  # nobody popped

        # Sibling scope: starts clean.
        with next_agent_span(metric_collection="fresh"):
            assert pop_pending_for("agent") == {"metric_collection": "fresh"}

    def test_drain_visible_across_asyncio_sub_context(self):
        """Regression: ``Agent.run_sync(...)`` calls ``asyncio.run(...)``
        which creates a new asyncio context that inherits a SNAPSHOT of
        contextvars. A naive ``ContextVar.set(None)`` from inside that
        snapshot would not propagate back, letting a second consumer in
        the parent context re-consume the same value.

        This test simulates the failure mode by running the consumer
        inside ``asyncio.run`` and verifying that the second consumer
        in the OUTER context sees the slot already drained.
        """
        import asyncio

        with next_agent_span(metric_collection="only-once"):

            async def _consume():
                return pop_pending_for("agent")

            # First consumer runs inside asyncio.run — same trick
            # ``Agent.run_sync`` plays internally.
            first = asyncio.run(_consume())
            assert first == {"metric_collection": "only-once"}

            # Second consumer in the outer ``with`` context. Must see
            # an empty dict because the asyncio sub-context's drain
            # mutated the shared ``_PendingSlot``.
            second = pop_pending_for("agent")
            assert second == {}

    def test_other_typed_helpers_each_use_their_own_slot(self):
        """Smoke test that ``next_tool_span`` / ``next_retriever_span``
        wire up to their respective slots (not the base/agent/llm
        slots)."""
        with next_tool_span(description="foo"):
            assert pop_pending_for("tool") == {"description": "foo"}
            assert pop_pending_for("agent") == {}

        with next_retriever_span(top_k=3, embedder="ada-002"):
            assert pop_pending_for("retriever") == {
                "top_k": 3,
                "embedder": "ada-002",
            }


# ---------------------------------------------------------------------------
# apply_pending_to_span — placeholder mutation behavior.
# ---------------------------------------------------------------------------


def _make_placeholder(cls=BaseSpan, **kw) -> BaseSpan:
    """Helper to build a minimal placeholder for applier tests."""
    base_kwargs = {
        "uuid": "u-1",
        "trace_uuid": "t-1",
        "status": TraceSpanStatus.IN_PROGRESS,
        "start_time": 0.0,
    }
    if cls is AgentSpan:
        base_kwargs.setdefault("name", "agent")
    base_kwargs.update(kw)
    return cls(**base_kwargs)


class TestApplyPendingToSpan:
    def test_empty_payload_is_noop(self):
        span = _make_placeholder()
        apply_pending_to_span(span, {})
        # Nothing changed — sanity.
        assert span.metric_collection is None
        assert span.metadata is None

    def test_base_field_setattr(self):
        span = _make_placeholder()
        apply_pending_to_span(
            span,
            {"metric_collection": "mc", "metadata": {"k": "v"}, "name": "n"},
        )
        assert span.metric_collection == "mc"
        assert span.metadata == {"k": "v"}
        assert span.name == "n"

    def test_agent_specific_fields_apply_only_to_agent_span(self):
        agent = _make_placeholder(AgentSpan)
        apply_pending_to_span(
            agent,
            {"available_tools": ["a", "b"], "agent_handoffs": ["h1"]},
        )
        assert agent.available_tools == ["a", "b"]
        assert agent.agent_handoffs == ["h1"]

    def test_cross_type_keys_silently_dropped(self):
        """Applier is hasattr-guarded: typed kwargs that don't apply to
        the placeholder's class are silently skipped instead of raising."""
        base = _make_placeholder()  # plain BaseSpan, no model/embedder/etc.
        apply_pending_to_span(
            base,
            {
                "model": "gpt-4",  # llm-only
                "embedder": "ada",  # retriever-only
                "available_tools": ["x"],  # agent-only
                "metric_collection": "shared",  # base — should land
            },
        )
        # Only the shared base field landed.
        assert base.metric_collection == "shared"
        # Cross-type keys did not raise and did not phantom-attribute.
        assert not hasattr(base, "model")
        assert not hasattr(base, "embedder")

    def test_test_case_unpacking_then_individual_fields_override(self):
        """``test_case`` is unpacked first, then individual base fields
        applied — so individual kwargs override the test_case's
        equivalent fields. Mirrors ``update_current_span(...)``'s order
        of operations (test_case first, then ``input``/``output``/etc.).
        Asserting this so the contract doesn't quietly flip."""
        from deepeval.test_case.llm_test_case import LLMTestCase

        span = _make_placeholder()
        tc = LLMTestCase(
            input="tc-input",
            actual_output="tc-output",
            expected_output="tc-expected",
        )
        apply_pending_to_span(
            span,
            {
                "test_case": tc,
                "input": "individual-input",  # overrides tc.input
                "output": "individual-output",  # overrides tc.actual_output
                # expected_output not overridden — falls through to tc.
            },
        )
        assert span.input == "individual-input"
        assert span.output == "individual-output"
        assert span.expected_output == "tc-expected"


# ---------------------------------------------------------------------------
# next_*_span ↔ SpanInterceptor wiring: end-to-end behavior.
#
# Verifies that ``with next_*_span(...)`` defaults actually land on the
# placeholder pushed by ``SpanInterceptor._push_span_context`` and end
# up in the OTel ``confident.span.*`` attrs after on_end.
# ---------------------------------------------------------------------------


def _make_agent_span_mock(agent_name="agent_x"):
    """Mock a pydantic-ai-style root agent span (operation_name=invoke_agent
    so SpanInterceptor classifies it as agent)."""
    return _make_mock_span(operation_name="invoke_agent", agent_name=agent_name)


class TestNextSpanInterceptorIntegration:
    def test_next_agent_span_metric_collection_lands_on_otel_attrs(self):
        """``with next_agent_span(metric_collection=...)`` is consumed by
        the interceptor's ``_push_span_context`` for the agent span and
        emitted as ``confident.span.metric_collection``."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_agent_span_mock()

        with next_agent_span(metric_collection="agent_metrics_v1"):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "agent_metrics_v1"
        )

    def test_next_agent_span_consumed_only_by_first_agent_span(self):
        """One-shot semantics through the interceptor: a second agent
        span inside the same ``with`` block does NOT inherit."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        first = _make_agent_span_mock("agent_one")
        second = _make_agent_span_mock("agent_two")

        with next_agent_span(metric_collection="only-first"):
            interceptor.on_start(first, None)
            interceptor.on_end(first)

            interceptor.on_start(second, None)
            interceptor.on_end(second)

        assert (
            first.attributes.get("confident.span.metric_collection")
            == "only-first"
        )
        assert second.attributes.get("confident.span.metric_collection") is None

    def test_next_agent_span_does_not_affect_non_agent_span(self):
        """Typed slot is NOT consumed by spans of a different type. An
        LLM span fired inside ``with next_agent_span(...)`` should pop
        nothing from the agent slot. The agent slot is still available
        for a subsequent agent span."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        llm_span = _make_mock_span(operation_name="chat")
        agent_span = _make_agent_span_mock()

        with next_agent_span(metric_collection="agent-only"):
            interceptor.on_start(llm_span, None)
            interceptor.on_end(llm_span)

            interceptor.on_start(agent_span, None)
            interceptor.on_end(agent_span)

        assert (
            llm_span.attributes.get("confident.span.metric_collection") is None
        )
        assert (
            agent_span.attributes.get("confident.span.metric_collection")
            == "agent-only"
        )

    def test_next_agent_span_metadata_lands_on_agent_placeholder(self):
        """``next_agent_span(metadata=...)`` flows through to the
        placeholder and is serialized to ``confident.span.metadata`` at
        on_end. Verifies non-metric_collection base kwargs make it
        through the consumption + serialization pipeline."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_agent_span_mock()

        with next_agent_span(metadata={"flow_check": "ok", "phase": "init"}):
            interceptor.on_start(span, None)
            # Placeholder is what next_agent_span wrote to.
            placeholder = current_span_context.get()
            assert placeholder.metadata == {
                "flow_check": "ok",
                "phase": "init",
            }
            interceptor.on_end(span)

        assert json.loads(span.attributes["confident.span.metadata"]) == {
            "flow_check": "ok",
            "phase": "init",
        }

    def test_update_current_span_overrides_next_agent_span_after_creation(
        self,
    ):
        """Last-write-wins: ``next_agent_span`` sets the floor at
        on_start; later ``update_current_span(...)`` calls (e.g. from
        inside a tool body that walks up to the agent placeholder)
        overwrite. Mirrors the trace-level precedence story."""
        settings = _make_settings()
        interceptor = SpanInterceptor(settings)
        span = _make_agent_span_mock()

        with next_agent_span(metric_collection="from-wrapper"):
            interceptor.on_start(span, None)
            update_current_span(metric_collection="from-update")
            interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "from-update"
        )
