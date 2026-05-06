"""Unit tests for ``AgentCoreSpanInterceptor`` (AgentCore OTel integration).

Mirrors the Pydantic AI test suite at
``tests/test_integrations/test_pydanticai/test_span_interceptor.py``.
Verifies the OTel POC pattern was correctly applied to AgentCore:

  - Trace-level reads from ``current_trace_context`` (with
    ``AgentCoreInstrumentationSettings`` defaults as fallback).
  - Span-context push/pop: ``current_span_context`` carries a
    ``BaseSpan`` placeholder for the OTel span's lifetime so
    ``update_current_span(...)`` from inside a Strands ``@tool`` body
    lands on the placeholder, then is serialized back into
    ``confident.span.*`` OTel attrs at on_end.
  - Implicit trace placeholder push for bare callers (no enclosing
    ``@observe`` / ``with trace(...)``) so
    ``update_current_trace(...)`` from inside a tool body works.
  - Parent bridge: ``confident.span.parent_uuid`` stamped on OTel roots
    when an enclosing real deepeval span is present.
  - ``next_*_span(...)`` payloads consumed at on_start; component-level
    metrics survive OTel transport via ``stash_pending_metrics``.
  - Removed top-level kwargs raise ``TypeError``.

These tests do NOT require AWS credentials or the ``bedrock_agentcore``
/ ``strands`` packages — they drive the interceptor with synthetic OTel
spans built from ``MagicMock``.
"""

from __future__ import annotations

import json
from itertools import count
from unittest.mock import MagicMock, patch

import pytest

from deepeval.integrations.agentcore.instrumentator import (
    AgentCoreInstrumentationSettings,
    AgentCoreSpanInterceptor,
)
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    next_agent_span,
    next_llm_span,
    next_tool_span,
    update_current_span,
    update_current_trace,
)
from deepeval.tracing.trace_context import trace


_span_id_counter = count(start=1)
_trace_id_counter = count(start=1)


def _make_mock_span(
    *,
    operation_name: str | None = None,
    agent_name: str | None = None,
    tool_name: str | None = None,
    span_name: str = "",
    parent: object | None = None,
):
    """Mock OTel span shaped to match ``AgentCoreSpanInterceptor``'s
    expectations.

    Mirrors the OTel SDK invariant that ``Span.attributes`` is a view
    over the same underlying ``_attributes`` mapping — so writes via
    either ``set_attribute(...)`` or direct ``_attributes[k] = v``
    (used by ``_set_attr_post_end`` to bypass the ended-span guard) are
    observable via ``span.attributes.get(...)``.

    AgentCore-specific differences from the Pydantic AI mock:
      - ``span.name`` is a plain string (the classifier calls
        ``.lower()`` on it). Default empty so the heuristic span-name
        fallback in ``_classify_span`` doesn't fire spuriously.
      - ``span.events`` defaults to ``[]`` so ``_extract_messages`` /
        ``_extract_tool_calls`` iterate cleanly.
    """
    span = MagicMock()
    backing: dict = {}
    span._attributes = backing
    span.attributes = backing
    span.name = span_name
    span.events = []
    span.start_time = None  # forces _push_span_context to use perf_counter()
    span.parent = parent  # None → root span
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
    return span


def _make_settings(**kwargs):
    """Return a minimal mock ``AgentCoreInstrumentationSettings``.

    Only fields ``AgentCoreSpanInterceptor`` actually reads. ``spec=[]``
    disallows auto-attrs so a typo on the interceptor side surfaces as
    AttributeError rather than a silent ``MagicMock``.

    Settings carries only trace-level fields (no per-span
    metric_collection / prompt / metrics) — span-level configuration
    is a runtime concern (``update_current_span(...)`` from inside a
    tool body, or ``with next_*_span(...)`` at the call site).
    """
    settings = MagicMock(spec=[])
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


def _make_agent_span_mock(agent_name: str = "agent_x"):
    """Mock a Strands-style root agent span (operation_name=invoke_agent
    so AgentCoreSpanInterceptor classifies it as agent)."""
    return _make_mock_span(operation_name="invoke_agent", agent_name=agent_name)


# ---------------------------------------------------------------------------
# Trace-context reads — settings fallback + runtime override.
# ---------------------------------------------------------------------------


class TestTraceContextReads:
    def test_uses_settings_when_no_trace_context(self):
        """Falls back to settings when current_trace_context is None."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings(
                thread_id="settings-thread",
                name="settings-name",
                metadata={"source": "settings"},
            )
            interceptor = AgentCoreSpanInterceptor(settings)
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
        settings = _make_settings(
            thread_id="settings-thread",
            name="settings-name",
        )
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_mock_span()

        with trace(thread_id="ctx-thread", name="ctx-name"):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.thread_id") == "ctx-thread"
        assert span.attributes.get("confident.trace.name") == "ctx-name"

    def test_metadata_is_merged_with_context_winning(self):
        settings = _make_settings(
            metadata={"base_key": "base_val", "shared_key": "from_settings"},
        )
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_mock_span()

        with trace(metadata={"ctx_key": "ctx_val", "shared_key": "from_ctx"}):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        result = json.loads(span.attributes["confident.trace.metadata"])
        assert result["base_key"] == "base_val"
        assert result["ctx_key"] == "ctx_val"
        assert result["shared_key"] == "from_ctx"

    def test_update_current_trace_after_on_start_lands_on_otel_attrs(self):
        """Trace attrs are snapshotted FRESH at on_end, not on_start.

        Regression guard for the at-on_start asymmetry: if a downstream
        caller mutates the active trace via ``update_current_trace``
        AFTER the OTel span's ``on_start`` has fired (e.g. from inside
        a Strands ``@tool`` body), the new values must still land on
        ``confident.trace.*`` when ``on_end`` runs.
        """
        settings = _make_settings(name="settings-name")
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_mock_span()

        with trace(name="initial-name"):
            interceptor.on_start(span, None)

            update_current_trace(
                name="updated-name",
                user_id="updated-user",
                metadata={"phase": "post-start"},
            )

            interceptor.on_end(span)

        assert span.attributes.get("confident.trace.name") == "updated-name"
        assert span.attributes.get("confident.trace.user_id") == "updated-user"
        assert json.loads(span.attributes["confident.trace.metadata"]) == {
            "phase": "post-start"
        }

    def test_trace_metric_collection_resolution_order(self):
        settings = _make_settings(metric_collection="settings-mc")
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_mock_span()

        with trace(metric_collection="ctx-mc"):
            interceptor.on_start(span, None)
            interceptor.on_end(span)

        assert (
            span.attributes.get("confident.trace.metric_collection") == "ctx-mc"
        )


# ---------------------------------------------------------------------------
# Span placeholder push / pop on current_span_context.
# ---------------------------------------------------------------------------


class TestSpanContextPushPop:
    def test_current_span_context_set_during_span_lifetime(self):
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_mock_span()

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
        interceptor = AgentCoreSpanInterceptor(settings)
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
        """``update_current_span(metric_collection=...)`` from inside a
        Strands ``@tool`` body lands on the tool span's OTel attrs.
        Direct analog of the ``special_tool`` flow in
        ``apps/agentcore_eval_app.py``."""
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_mock_span()

        interceptor.on_start(span, None)
        update_current_span(metric_collection="runtime-collection")
        interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "runtime-collection"
        )

    def test_nested_spans_lifo_pop_restores_parent_placeholder(self):
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        outer = _make_mock_span()
        inner = _make_mock_span(parent=MagicMock())

        interceptor.on_start(outer, None)
        outer_placeholder = current_span_context.get()

        interceptor.on_start(inner, None)
        inner_placeholder = current_span_context.get()
        assert inner_placeholder is not outer_placeholder

        interceptor.on_end(inner)
        assert current_span_context.get() is outer_placeholder

        interceptor.on_end(outer)


# ---------------------------------------------------------------------------
# Implicit trace placeholder push for bare ``invoke(...)`` callers.
# ---------------------------------------------------------------------------


class TestImplicitTraceContext:
    """Symmetric to ``TestSpanContextPushPop`` but at the trace level.
    The interceptor pushes an implicit ``Trace`` placeholder onto
    ``current_trace_context`` for the OTel root span's lifetime so
    ``update_current_trace(...)`` from inside Strands tools / nested
    helpers can mutate something. The placeholder is tagged
    ``is_otel_implicit=True`` so ``ContextAwareSpanProcessor`` keeps
    routing to OTLP for those callers.
    """

    def test_root_span_pushes_implicit_trace_when_no_user_context(self):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = AgentCoreSpanInterceptor(settings)
            root = _make_mock_span()

            interceptor.on_start(root, None)
            during = current_trace_context.get()

            assert during is not None
            assert getattr(during, "is_otel_implicit", False) is True

            interceptor.on_end(root)
            assert current_trace_context.get() is None
        finally:
            current_trace_context.reset(token)

    def test_does_not_overwrite_user_pushed_trace_context(self):
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        root = _make_mock_span()

        with trace() as user_trace:
            assert getattr(user_trace, "is_otel_implicit", False) is False

            interceptor.on_start(root, None)
            during = current_trace_context.get()

            assert during is user_trace
            assert getattr(during, "is_otel_implicit", False) is False

            interceptor.on_end(root)

            assert current_trace_context.get() is user_trace

    def test_child_span_does_not_push_its_own_placeholder(self):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = AgentCoreSpanInterceptor(settings)
            root = _make_mock_span()
            child = _make_mock_span(parent=MagicMock())

            interceptor.on_start(root, None)
            implicit = current_trace_context.get()
            assert implicit is not None

            interceptor.on_start(child, None)
            assert current_trace_context.get() is implicit

            interceptor.on_end(child)
            assert current_trace_context.get() is implicit

            interceptor.on_end(root)
            assert current_trace_context.get() is None
        finally:
            current_trace_context.reset(token)

    def test_update_current_trace_in_implicit_context_lands_on_otel_attrs(
        self,
    ):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = AgentCoreSpanInterceptor(settings)
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
# Parent bridge: confident.span.parent_uuid stamping for OTel roots
# inside an enclosing deepeval (real, non-implicit) span.
# ---------------------------------------------------------------------------


class TestParentBridge:
    def test_stamps_parent_uuid_when_enclosed_in_deepeval_span(self):
        """When a real deepeval span is on ``current_span_context`` and
        the OTel span is a root (no native parent), the interceptor
        stamps ``confident.span.parent_uuid`` so the exporter can
        re-parent the OTel root onto the deepeval span instead of
        emitting it as a sibling.
        """
        from deepeval.tracing.types import BaseSpan, TraceSpanStatus

        outer = BaseSpan(
            uuid="deepeval-outer-uuid",
            trace_uuid="deepeval-trace-uuid",
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=0.0,
        )
        token = current_span_context.set(outer)
        try:
            settings = _make_settings()
            interceptor = AgentCoreSpanInterceptor(settings)
            root = _make_mock_span()  # parent=None makes it a root

            interceptor.on_start(root, None)
            interceptor.on_end(root)

            assert (
                root.attributes.get("confident.span.parent_uuid")
                == "deepeval-outer-uuid"
            )
        finally:
            current_span_context.reset(token)

    def test_no_parent_uuid_when_otel_span_has_native_parent(self):
        """OTel children already have a real parent_id pointing into
        the same OTel trace — no need to bridge."""
        from deepeval.tracing.types import BaseSpan, TraceSpanStatus

        outer = BaseSpan(
            uuid="deepeval-outer-uuid",
            trace_uuid="deepeval-trace-uuid",
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=0.0,
        )
        token = current_span_context.set(outer)
        try:
            settings = _make_settings()
            interceptor = AgentCoreSpanInterceptor(settings)
            child = _make_mock_span(parent=MagicMock())

            interceptor.on_start(child, None)
            interceptor.on_end(child)

            assert "confident.span.parent_uuid" not in child.attributes
        finally:
            current_span_context.reset(token)


# ---------------------------------------------------------------------------
# next_*_span(...) consumption + stash_pending_metrics gating.
# ---------------------------------------------------------------------------


class TestNextSpanInterceptorIntegration:
    def test_next_agent_span_metric_collection_lands_on_otel_attrs(self):
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
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
        interceptor = AgentCoreSpanInterceptor(settings)
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
        nothing from the agent slot."""
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
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

    def test_next_tool_span_metric_collection_lands_on_tool_otel_attrs(self):
        """Mirrors the ``test_tool_metric_collection`` flow in test_sync.py
        — ``with next_tool_span(metric_collection=...)`` sets the value
        on the FIRST tool span emitted inside the block."""
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        tool_span = _make_mock_span(
            operation_name="execute_tool", tool_name="calculate"
        )

        with next_tool_span(metric_collection="calculator-metrics"):
            interceptor.on_start(tool_span, None)
            interceptor.on_end(tool_span)

        assert (
            tool_span.attributes.get("confident.span.metric_collection")
            == "calculator-metrics"
        )

    def test_update_current_span_overrides_next_agent_span_after_creation(
        self,
    ):
        """Last-write-wins: ``next_agent_span`` sets the floor at
        on_start; later ``update_current_span(...)`` (e.g. from inside
        a tool body) overwrites."""
        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_agent_span_mock()

        with next_agent_span(metric_collection="from-wrapper"):
            interceptor.on_start(span, None)
            update_current_span(metric_collection="from-update")
            interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "from-update"
        )

    def test_next_agent_span_metrics_stashed_when_evaluating(self):
        """``with next_agent_span(metrics=[...])`` populates the
        placeholder; at on_end the interceptor calls
        ``stash_pending_metrics`` so ``ConfidentSpanExporter`` can
        re-attach the ``BaseMetric`` instances after rebuilding the
        span (they don't fit in OTel primitives-only attrs).

        Gated on ``trace_manager.is_evaluating`` to keep the registry
        from growing in production paths.
        """
        from deepeval.metrics import AnswerRelevancyMetric

        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_agent_span_mock()
        metric = AnswerRelevancyMetric()

        with patch(
            "deepeval.integrations.agentcore.instrumentator."
            "stash_pending_metrics"
        ) as stash, patch(
            "deepeval.integrations.agentcore.instrumentator." "trace_manager"
        ) as fake_tm:
            fake_tm.is_evaluating = True
            with next_agent_span(metrics=[metric]):
                interceptor.on_start(span, None)
                interceptor.on_end(span)

        stash.assert_called_once()
        # First positional arg = uuid (16-char hex), second = metrics list.
        args, _ = stash.call_args
        assert isinstance(args[0], str) and len(args[0]) == 16
        assert args[1] == [metric]

    def test_next_agent_span_metrics_not_stashed_outside_eval_mode(self):
        """In production paths (``is_evaluating=False``) the metrics
        overlay would leak — gate prevents the stash."""
        from deepeval.metrics import AnswerRelevancyMetric

        settings = _make_settings()
        interceptor = AgentCoreSpanInterceptor(settings)
        span = _make_agent_span_mock()
        metric = AnswerRelevancyMetric()

        with patch(
            "deepeval.integrations.agentcore.instrumentator."
            "stash_pending_metrics"
        ) as stash, patch(
            "deepeval.integrations.agentcore.instrumentator." "trace_manager"
        ) as fake_tm:
            fake_tm.is_evaluating = False
            with next_agent_span(metrics=[metric]):
                interceptor.on_start(span, None)
                interceptor.on_end(span)

        stash.assert_not_called()


# ---------------------------------------------------------------------------
# Removed kwargs: settings + instrument_agentcore signature.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwarg",
    [
        "is_test_mode",
        "agent_metric_collection",
        "llm_metric_collection",
        "tool_metric_collection_map",
        "trace_metric_collection",
        "agent_metrics",
        "confident_prompt",
    ],
)
def test_removed_kwargs_raise_typeerror_on_settings(kwarg):
    """Span-level kwargs were removed in the OTel POC migration. Each
    must raise ``TypeError`` on construction so callers see exactly
    which kwarg to migrate."""
    with pytest.raises(TypeError) as exc:
        AgentCoreInstrumentationSettings(api_key="dummy", **{kwarg: object()})

    # The error message names the removed kwarg, so a future expansion
    # of ``_REMOVED_KWARGS`` doesn't accidentally swallow it.
    assert kwarg in str(exc.value)


@pytest.mark.parametrize(
    "kwarg",
    [
        "is_test_mode",
        "agent_metric_collection",
        "llm_metric_collection",
        "tool_metric_collection_map",
        "trace_metric_collection",
        "agent_metrics",
        "confident_prompt",
    ],
)
def test_removed_kwargs_raise_typeerror_on_instrument_agentcore(kwarg):
    """Same guard at the ``instrument_agentcore(...)`` entry point —
    catches callers that bypass the settings constructor."""
    from deepeval.integrations.agentcore import instrument_agentcore

    with pytest.raises(TypeError) as exc:
        instrument_agentcore(api_key="dummy", **{kwarg: object()})

    assert kwarg in str(exc.value)


# ---------------------------------------------------------------------------
# Optional Confident AI api_key — must NOT be required.
# ---------------------------------------------------------------------------


def test_settings_no_api_key_does_not_raise(monkeypatch):
    """Constructor must succeed when no api_key is supplied or in env.

    The OTel pipeline still wires up locally — only the outbound auth
    header is gated on a key being present (handled in
    ``ContextAwareSpanProcessor``, not the settings constructor).
    """
    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)
    instance = AgentCoreInstrumentationSettings()
    assert instance is not None
    assert instance.api_key is None
