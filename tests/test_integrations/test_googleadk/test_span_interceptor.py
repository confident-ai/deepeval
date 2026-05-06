"""Unit tests for ``OpenInferenceSpanInterceptor`` driven by Google-ADK-shaped spans.

Mirrors ``tests/test_integrations/test_agentcore/test_span_interceptor.py``
(itself a port of the Pydantic AI suite). The interceptor under test is
shared across every OpenInference-backed integration — Google ADK is the
first user of it on the deepeval side, so this file is the canonical
synthetic-span coverage.

What this file verifies on the OpenInference span interceptor:

  - Trace-level reads from ``current_trace_context`` with
    ``OpenInferenceInstrumentationSettings`` defaults as fallback, FRESH
    resolution at on_end (so ``update_current_trace(...)`` from inside
    a tool body still lands), and metadata merge with context winning.
  - Span placeholder push/pop on ``current_span_context`` so
    ``update_current_span(...)`` from anywhere in the call stack
    serializes back to ``confident.span.*`` at on_end.
  - Implicit ``Trace(is_otel_implicit=True)`` push for bare ADK callers
    (no enclosing ``@observe`` / ``with trace(...)``) so
    ``update_current_trace(...)`` works without a user-pushed context.
  - Parent bridge: ``confident.span.parent_uuid`` stamped on OTel roots
    enclosed in a real (non-implicit) deepeval span.
  - ``next_*_span(...)`` consumption at on_start; component-level
    ``BaseMetric`` instances stashed via ``stash_pending_metrics``
    (gated on ``trace_manager.is_evaluating``).
  - Removed top-level kwargs (the OTel POC migration) raise
    ``TypeError`` on both ``OpenInferenceInstrumentationSettings`` and
    ``instrument_google_adk``.
  - OpenInference framework-attr extraction:
    ``openinference.span.kind`` → ``confident.span.type``,
    ``llm.input_messages.{idx}.message.content`` → ``confident.span.input``,
    ``llm.output_messages.{idx}...`` → ``confident.span.output``,
    nested ``...tool_calls.{tc}.tool_call.function.{name,arguments}`` →
    ``confident.span.tools_called``, ``llm.token_count.{prompt,completion}``
    → ``confident.llm.{input,output}_token_count``,
    ``llm.model_name`` → ``confident.llm.model``,
    tool spans' ``tool.name`` / ``tool.parameters`` →
    ``confident.span.tools_called`` (1-element list) +
    ``confident.span.input``.

These tests do NOT require ``google-adk`` /
``openinference-instrumentation-google-adk`` — they drive the
interceptor with synthetic OTel spans built from ``MagicMock``.
"""

from __future__ import annotations

import json
from itertools import count
from unittest.mock import MagicMock, patch

import pytest

from deepeval.integrations.openinference.instrumentator import (
    OpenInferenceInstrumentationSettings,
    OpenInferenceSpanInterceptor,
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
    span_kind: str | None = None,
    agent_name: str | None = None,
    tool_name: str | None = None,
    span_name: str = "",
    parent: object | None = None,
    extra_attrs: dict | None = None,
):
    """Mock OTel span shaped to match ``OpenInferenceSpanInterceptor``'s
    expectations.

    Mirrors the OTel SDK invariant that ``Span.attributes`` is a view
    over the same underlying ``_attributes`` mapping — so writes via
    either ``set_attribute(...)`` or direct ``_attributes[k] = v``
    (used by ``_set_attr_post_end`` to bypass the ended-span guard) are
    observable via ``span.attributes.get(...)``.

    OpenInference / Google-ADK-specific differences from the
    AgentCore mock:

      - Classification reads ``openinference.span.kind`` (uppercased)
        instead of ``gen_ai.operation.name``. Recognized values:
        ``"AGENT"`` / ``"CHAIN"`` → agent, ``"LLM"`` → llm,
        ``"TOOL"`` → tool, ``"RETRIEVER"`` → retriever; anything else
        → ``"custom"``; missing → ``None`` (interceptor leaves it alone).
      - Agent / tool name come from ``agent.name`` / ``tool.name``
        (no ``gen_ai.`` prefix).
      - ``span.name`` is a plain string (used as the fallback for
        ``_get_agent_name`` / ``_get_tool_name``). Default empty so
        the fallback doesn't fire spuriously.
      - ``span.events`` defaults to ``[]`` for parity with the
        AgentCore mock; the OpenInference interceptor doesn't read
        events directly but downstream attr extraction is event-free.
    """
    span = MagicMock()
    backing: dict = {}
    span._attributes = backing
    span.attributes = backing
    span.name = span_name
    span.events = []
    span.start_time = None  # forces _push_span_context to use perf_counter()
    span.parent = parent  # None → root span
    if span_kind:
        backing["openinference.span.kind"] = span_kind
    if agent_name:
        backing["agent.name"] = agent_name
    if tool_name:
        backing["tool.name"] = tool_name
    if extra_attrs:
        backing.update(extra_attrs)
    span.set_attribute.side_effect = lambda k, v: backing.__setitem__(k, v)
    span.get_span_context.return_value = MagicMock(
        trace_id=next(_trace_id_counter),
        span_id=next(_span_id_counter),
    )
    return span


def _make_settings(**kwargs):
    """Return a minimal mock ``OpenInferenceInstrumentationSettings``.

    ``spec=[]`` disallows auto-attrs so a typo on the interceptor side
    surfaces as ``AttributeError`` rather than a silent ``MagicMock``.

    Settings carries only trace-level fields (no per-span
    metric_collection / prompt / metrics) — span-level config is a
    runtime concern (``update_current_span(...)`` from inside a tool
    body, or ``with next_*_span(...)`` at the call site).
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
    """Mock an OpenInference-shaped root agent span (kind=AGENT)."""
    return _make_mock_span(span_kind="AGENT", agent_name=agent_name)


def _make_tool_span_mock(tool_name: str = "calculate"):
    """Mock an OpenInference-shaped tool span (kind=TOOL)."""
    return _make_mock_span(span_kind="TOOL", tool_name=tool_name)


def _make_llm_span_mock():
    """Mock an OpenInference-shaped LLM span (kind=LLM)."""
    return _make_mock_span(span_kind="LLM")


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
            interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        an ADK tool body), the new values must still land on
        ``confident.trace.*`` when ``on_end`` runs.
        """
        settings = _make_settings(name="settings-name")
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        """``update_current_span(metric_collection=...)`` from inside an
        ADK tool body lands on the tool span's OTel attrs. Direct analog
        of the ``special_tool`` flow in ``apps/googleadk_eval_app.py``."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_tool_span_mock("special_tool")

        interceptor.on_start(span, None)
        update_current_span(metric_collection="runtime-collection")
        interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.metric_collection")
            == "runtime-collection"
        )

    def test_nested_spans_lifo_pop_restores_parent_placeholder(self):
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
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
# Implicit trace placeholder push for bare ADK callers.
# ---------------------------------------------------------------------------


class TestImplicitTraceContext:
    """Symmetric to ``TestSpanContextPushPop`` but at the trace level.
    The interceptor pushes an implicit ``Trace`` placeholder onto
    ``current_trace_context`` for the OTel root span's lifetime so
    ``update_current_trace(...)`` from inside ADK tools / nested
    helpers can mutate something. The placeholder is tagged
    ``is_otel_implicit=True`` so ``ContextAwareSpanProcessor`` keeps
    routing to OTLP for those callers.
    """

    def test_root_span_pushes_implicit_trace_when_no_user_context(self):
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
            interceptor = OpenInferenceSpanInterceptor(settings)
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
            interceptor = OpenInferenceSpanInterceptor(settings)
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
            interceptor = OpenInferenceSpanInterceptor(settings)
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
            interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
        llm_span = _make_llm_span_mock()
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
        interceptor = OpenInferenceSpanInterceptor(settings)
        tool_span = _make_tool_span_mock("calculate")

        with next_tool_span(metric_collection="calculator-metrics"):
            interceptor.on_start(tool_span, None)
            interceptor.on_end(tool_span)

        assert (
            tool_span.attributes.get("confident.span.metric_collection")
            == "calculator-metrics"
        )

    def test_next_llm_span_metric_collection_lands_on_llm_otel_attrs(self):
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        llm_span = _make_llm_span_mock()

        with next_llm_span(metric_collection="llm_metrics_v1"):
            interceptor.on_start(llm_span, None)
            interceptor.on_end(llm_span)

        assert (
            llm_span.attributes.get("confident.span.metric_collection")
            == "llm_metrics_v1"
        )

    def test_update_current_span_overrides_next_agent_span_after_creation(
        self,
    ):
        """Last-write-wins: ``next_agent_span`` sets the floor at
        on_start; later ``update_current_span(...)`` (e.g. from inside
        a tool body) overwrites."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
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
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_agent_span_mock()
        metric = AnswerRelevancyMetric()

        with patch(
            "deepeval.integrations.openinference.instrumentator."
            "stash_pending_metrics"
        ) as stash, patch(
            "deepeval.integrations.openinference.instrumentator.trace_manager"
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
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_agent_span_mock()
        metric = AnswerRelevancyMetric()

        with patch(
            "deepeval.integrations.openinference.instrumentator."
            "stash_pending_metrics"
        ) as stash, patch(
            "deepeval.integrations.openinference.instrumentator.trace_manager"
        ) as fake_tm:
            fake_tm.is_evaluating = False
            with next_agent_span(metrics=[metric]):
                interceptor.on_start(span, None)
                interceptor.on_end(span)

        stash.assert_not_called()


# ---------------------------------------------------------------------------
# OpenInference framework-attr extraction (the bit that's specific to
# this interceptor — AgentCore reads gen_ai.* / Strands events instead).
# ---------------------------------------------------------------------------


class TestFrameworkAttrExtraction:
    """Verifies the ``_serialize_framework_attrs`` path: classification,
    flattened message extraction, tool-call extraction (Scenario A:
    span IS a tool, Scenario B: tool calls embedded in an LLM output
    message), token counts, and model name. Framework attrs run with
    ``setdefault`` semantics — the placeholder serializer ran first
    so ``update_current_span(...)`` writes win over framework writes."""

    def test_agent_span_kind_lands_as_confident_span_type_agent(self):
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_agent_span_mock("planner")

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert span.attributes.get("confident.span.type") == "agent"
        assert span.attributes.get("confident.span.name") == "planner"

    def test_chain_span_kind_classified_as_agent(self):
        """OpenInference uses CHAIN for orchestration nodes that look
        agent-shaped to deepeval — both flow into AgentSpan."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(span_kind="CHAIN", agent_name="root_chain")

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert span.attributes.get("confident.span.type") == "agent"

    def test_llm_span_kind_lands_as_confident_span_type_llm(self):
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_llm_span_mock()

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert span.attributes.get("confident.span.type") == "llm"

    def test_tool_span_kind_lands_as_confident_span_type_tool(self):
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_tool_span_mock("calculate")

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert span.attributes.get("confident.span.type") == "tool"
        assert span.attributes.get("confident.span.name") == "calculate"

    def test_unknown_span_kind_classified_as_custom(self):
        """Anything that's not AGENT / CHAIN / LLM / TOOL / RETRIEVER
        falls through to ``custom`` so non-standard OpenInference
        instrumentors still get represented."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(span_kind="GUARDRAIL")

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert span.attributes.get("confident.span.type") == "custom"

    def test_missing_span_kind_leaves_type_unset(self):
        """Spans without ``openinference.span.kind`` are not
        OpenInference-emitted; the interceptor must not stamp a type
        on them so they don't get rebuilt as malformed deepeval spans."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span()  # no kind set

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert "confident.span.type" not in span.attributes

    def test_llm_span_extracts_flattened_input_output_messages(self):
        """OpenInference flattens chat history into
        ``llm.{input,output}_messages.{idx}.message.content``. The
        interceptor walks the indexes until a hole, takes the LAST
        seen content, and writes it to ``confident.span.{input,output}``.
        """
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(
            span_kind="LLM",
            extra_attrs={
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": "You are concise.",
                "llm.input_messages.1.message.role": "user",
                "llm.input_messages.1.message.content": "Hello?",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "Hi!",
            },
        )

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        # Last input message wins (assistant context normally trails
        # at output); for input we expect the latest user turn.
        assert span.attributes.get("confident.span.input") == "Hello?"
        assert span.attributes.get("confident.span.output") == "Hi!"

    def test_llm_span_extracts_token_counts_and_model_name(self):
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(
            span_kind="LLM",
            extra_attrs={
                "llm.token_count.prompt": 42,
                "llm.token_count.completion": 17,
                "llm.model_name": "gemini-2.0-flash",
            },
        )

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert span.attributes.get("confident.llm.input_token_count") == 42
        assert span.attributes.get("confident.llm.output_token_count") == 17
        assert span.attributes.get("confident.llm.model") == "gemini-2.0-flash"

    def test_llm_span_extracts_tool_calls_from_output_messages(self):
        """Scenario B: tool calls embedded inside an LLM output
        message via the flattened
        ``llm.output_messages.{idx}.message.tool_calls.{tc}.tool_call.function.{name,arguments}``
        attrs. The interceptor walks ``msg_idx`` outer × ``tc_idx``
        inner, JSON-parses ``arguments``, and emits a
        ``confident.span.tools_called`` JSON list of ``ToolCall``s.
        """
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(
            span_kind="LLM",
            extra_attrs={
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "",
                "llm.output_messages.0.message.tool_calls.0."
                "tool_call.function.name": "get_weather",
                "llm.output_messages.0.message.tool_calls.0."
                "tool_call.function.arguments": '{"city": "Tokyo"}',
                "llm.output_messages.0.message.tool_calls.1."
                "tool_call.function.name": "get_time",
                "llm.output_messages.0.message.tool_calls.1."
                "tool_call.function.arguments": '{"city": "Tokyo"}',
            },
        )

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        raw = span.attributes.get("confident.span.tools_called")
        assert raw is not None
        # Each entry is a ToolCall.model_dump_json() string.
        parsed = [json.loads(item) for item in raw]
        names = sorted(p["name"] for p in parsed)
        assert names == ["get_time", "get_weather"]
        for p in parsed:
            assert p["input_parameters"] == {"city": "Tokyo"}

    def test_tool_span_extracts_self_as_single_tool_call(self):
        """Scenario A: the span itself is a tool span (kind=TOOL),
        so the framework extractor builds a 1-element
        ``confident.span.tools_called`` from ``tool.name`` /
        ``tool.parameters`` and copies the parameters into
        ``confident.span.input`` for visibility."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(
            span_kind="TOOL",
            tool_name="get_weather",
            extra_attrs={
                "tool.parameters": '{"city": "Paris"}',
            },
        )

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        raw = span.attributes.get("confident.span.tools_called")
        assert raw is not None
        assert len(raw) == 1
        parsed = json.loads(raw[0])
        assert parsed["name"] == "get_weather"
        assert parsed["input_parameters"] == {"city": "Paris"}
        # ``confident.span.input`` was empty (no update_current_span);
        # framework path fills it from the tool params.
        assert json.loads(span.attributes["confident.span.input"]) == {
            "city": "Paris"
        }

    def test_agent_span_input_output_also_lands_on_trace_attrs(self):
        """Agent (root) spans surface their I/O onto
        ``confident.trace.{input,output}`` too so the trace card has
        prompt + response without re-walking spans server-side."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(
            span_kind="AGENT",
            agent_name="planner",
            extra_attrs={
                "input.value": "What's the weather in Tokyo?",
                "output.value": "Sunny, 72F.",
            },
        )

        interceptor.on_start(span, None)
        interceptor.on_end(span)

        assert (
            span.attributes.get("confident.span.input")
            == "What's the weather in Tokyo?"
        )
        assert span.attributes.get("confident.span.output") == "Sunny, 72F."
        assert (
            span.attributes.get("confident.trace.input")
            == "What's the weather in Tokyo?"
        )
        assert span.attributes.get("confident.trace.output") == "Sunny, 72F."

    def test_update_current_span_input_wins_over_framework_input(self):
        """Framework path uses ``setdefault`` semantics — when the
        placeholder serializer (which runs first) already stamped
        ``confident.span.input``, the framework path must not
        overwrite it. Regression guard for the layering order."""
        settings = _make_settings()
        interceptor = OpenInferenceSpanInterceptor(settings)
        span = _make_mock_span(
            span_kind="LLM",
            extra_attrs={
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "framework-input",
            },
        )

        interceptor.on_start(span, None)
        update_current_span(input="user-supplied-input")
        interceptor.on_end(span)

        assert (
            json.loads(span.attributes["confident.span.input"])
            == "user-supplied-input"
        )


# ---------------------------------------------------------------------------
# Removed kwargs: settings + instrument_google_adk signature.
# ---------------------------------------------------------------------------


_REMOVED_KWARGS = [
    "is_test_mode",
    "agent_metric_collection",
    "llm_metric_collection",
    "tool_metric_collection_map",
    "trace_metric_collection",
    "agent_metrics",
    "confident_prompt",
]


@pytest.mark.parametrize("kwarg", _REMOVED_KWARGS)
def test_removed_kwargs_raise_typeerror_on_settings(kwarg):
    """Span-level kwargs were removed in the OTel POC migration. Each
    must raise ``TypeError`` on construction so callers see exactly
    which kwarg to migrate."""
    with pytest.raises(TypeError) as exc:
        OpenInferenceInstrumentationSettings(
            api_key="dummy", **{kwarg: object()}
        )

    # The error message names the removed kwarg, so a future expansion
    # of ``_REMOVED_KWARGS`` doesn't accidentally swallow it.
    assert kwarg in str(exc.value)


@pytest.mark.parametrize("kwarg", _REMOVED_KWARGS)
def test_removed_kwargs_raise_typeerror_on_instrument_google_adk(kwarg):
    """Same guard at the ``instrument_google_adk(...)`` entry point —
    catches callers that bypass the settings constructor. The kwarg
    check fires BEFORE the GoogleADKInstrumentor import, so this test
    works without ``openinference-instrumentation-google-adk`` installed.
    """
    from deepeval.integrations.google_adk import instrument_google_adk

    with pytest.raises(TypeError) as exc:
        instrument_google_adk(api_key="dummy", **{kwarg: object()})

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
    instance = OpenInferenceInstrumentationSettings()
    assert instance is not None
    assert instance.api_key is None
