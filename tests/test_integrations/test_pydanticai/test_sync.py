"""
Sync PydanticAI Tests
All synchronous tests using deterministic settings.
"""

import os
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

from tests.test_integrations.test_pydanticai.apps.eval_app import (
    create_evals_agent,
    invoke_evals_agent,
)

# App imports
from tests.test_integrations.test_pydanticai.apps.pydanticai_simple_app import (
    create_simple_agent,
    invoke_simple_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_tool_app import (
    create_tool_agent,
    invoke_tool_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_metric_collection_app import (
    create_trace_metric_collection_agent,
    invoke_metric_collection_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_multiple_tools_app import (
    create_multiple_tools_agent,
    invoke_multiple_tools_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_next_span_app import (
    create_next_span_agent,
    invoke_with_next_llm_span,
    invoke_with_stacked_next_spans,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_modes_app import (
    create_enrichment_agent,
    create_modes_agent,
    invoke_in_observe_mode,
    invoke_in_with_trace_mode,
    invoke_with_tool_enrichment,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_isolation_app import (
    create_isolation_agent,
    make_distinct_requests,
    threaded_isolation_run,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    """
    Decorator that switches between generate and assert mode based on GENERATE_SCHEMAS env var.

    Args:
        schema_name: Name of the schema file (without path)
    """
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


# =============================================================================
# SIMPLE APP TESTS (LLM only, no tools)
# =============================================================================


class TestSimpleApp:
    """Tests for simple LLM-only PydanticAI agent."""

    @trace_test("pydanticai_simple_schema.json")
    def test_simple_greeting(self):
        """Test a simple greeting that returns a response."""
        agent = create_simple_agent(
            name="pydanticai-simple-test",
            tags=["pydanticai", "simple"],
            metadata={"test_type": "simple"},
            thread_id="simple-123",
            user_id="test-user",
        )

        result = invoke_simple_agent(
            "Say hello in exactly three words.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# TOOL APP TESTS (Agent with tool calling)
# =============================================================================


class TestToolApp:
    """Tests for PydanticAI agent with tool calling."""

    @trace_test("pydanticai_tool_schema.json")
    def test_tool_calculation(self):
        """Test a simple calculation using a tool."""
        agent = create_tool_agent(
            name="pydanticai-tool-test",
            tags=["pydanticai", "tool"],
            metadata={"test_type": "tool"},
            thread_id="tool-123",
            user_id="test-user",
        )

        result = invoke_tool_agent(
            "What is 7 multiplied by 8?",
            agent=agent,
        )

        assert result is not None
        assert "56" in result


# =============================================================================
# METRIC COLLECTION TESTS (Online evals)
# =============================================================================


class TestMetricCollectionApp:
    """Tests trace-level metric_collection set at runtime via
    ``update_current_trace(metric_collection=...)`` from inside a tool.
    Per-span metric_collection (agent / LLM / tool) is no longer a
    settings concern — set it at the call site via
    ``update_current_span(metric_collection=...)``.
    """

    @trace_test("pydanticai_trace_metric_collection_schema.json")
    def test_trace_metric_collection(self):
        """Test trace-level metric_collection set as a settings default."""
        agent = create_trace_metric_collection_agent(
            metric_collection="test-trace-metrics",
            name="pydanticai-trace-metric-test",
            tags=["pydanticai", "trace-metric-collection"],
            metadata={"test_type": "trace_metric_collection"},
            thread_id="trace-metric-123",
            user_id="test-user",
        )

        result = invoke_metric_collection_agent(
            "Say hello in exactly two words.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# MULTIPLE TOOLS TESTS
# =============================================================================


class TestMultipleToolsApp:
    """Tests for PydanticAI agent with multiple tools."""

    @trace_test("pydanticai_multiple_tools_weather_schema.json")
    def test_multiple_tools_weather_only(self):
        """Test calling get_weather tool when agent has multiple tools available."""
        agent = create_multiple_tools_agent(
            name="pydanticai-multiple-tools-weather",
            tags=["pydanticai", "multiple-tools", "weather"],
            metadata={"test_type": "multiple_tools_weather"},
            thread_id="multiple-tools-weather-123",
            user_id="test-user",
        )

        result = invoke_multiple_tools_agent(
            "Use the get_weather tool exactly once to get the weather in Tokyo.",
            agent=agent,
        )

        assert result is not None
        # Verify weather data is in response
        assert "72" in result or "sunny" in result.lower()

    @trace_test("pydanticai_multiple_tools_time_schema.json")
    def test_multiple_tools_time_only(self):
        """Test calling get_time tool when agent has multiple tools available."""
        agent = create_multiple_tools_agent(
            name="pydanticai-multiple-tools-time",
            tags=["pydanticai", "multiple-tools", "time"],
            metadata={"test_type": "multiple_tools_time"},
            thread_id="multiple-tools-time-123",
            user_id="test-user",
        )

        result = invoke_multiple_tools_agent(
            "Use the get_time tool exactly once to get the current time in London.",
            agent=agent,
        )

        assert result is not None
        # Verify time data is in response
        assert "7:00" in result or "GMT" in result

    @trace_test("pydanticai_parallel_tools_schema.json")
    def test_parallel_tool_calls(self):
        """Test calling both get_weather and get_time tools in parallel.

        PydanticAI supports parallel tool calls - when the LLM decides to call
        multiple tools, they are executed and results returned together.
        """
        agent = create_multiple_tools_agent(
            name="pydanticai-parallel-tools",
            tags=["pydanticai", "parallel-tools"],
            metadata={"test_type": "parallel_tools"},
            thread_id="parallel-tools-123",
            user_id="test-user",
        )

        result = invoke_multiple_tools_agent(
            "Use both the get_weather tool AND the get_time tool for Paris. "
            "Call both tools exactly once each.",
            agent=agent,
        )

        assert result is not None
        # Verify both weather and time data are in response
        # Weather should mention 62 or cloudy
        assert "62" in result or "cloudy" in result.lower()
        # Time should mention 8:00 or CET
        assert "8:00" in result or "CET" in result


# =============================================================================
# DEEPEVAL FEATURES TESTS
# =============================================================================


class TestDeepEvalFeatures:
    """Tests for DeepEval-specific trace-level settings + metadata."""

    @trace_test("pydanticai_features_sync.json")
    def test_full_features_sync(self):
        """Trace-level + agent-span-level features together. Trace
        ``metric_collection`` comes from settings (declarative default);
        agent-span ``metric_collection`` is staged via
        ``next_agent_span(...)`` since the user can't enter the agent
        span body."""
        agent = create_evals_agent(
            metric_collection="trace_metrics_override_v1",
            name="pydanticai-full-features-sync",
            tags=["pydanticai", "features", "sync"],
            metadata={"env": "testing", "priority": "high"},
            thread_id="thread-sync-features-001",
            user_id="user-sync-001",
        )

        result = invoke_evals_agent(
            "Use the special_tool to process 'Sync Data'",
            agent=agent,
            agent_metric_collection="agent_metrics_v1",
        )

        assert result is not None


# =============================================================================
# NEXT-SPAN STAGING TESTS (next_llm_span + stacked typed slots)
# =============================================================================


class TestNextSpanApp:
    """Schema-asserted coverage for ``with next_llm_span(...)`` and
    stacked ``with next_agent_span(...), next_llm_span(...)`` — the
    only mechanism for stamping LLM-span fields, since user code never
    runs inside an LLM span body. Mirrors scenarios 1 and 2 from
    ``pydantic_after_next_span.py``."""

    @trace_test("pydanticai_next_llm_only_schema.json")
    def test_next_llm_span_only(self):
        """``with next_llm_span(...)`` alone: LLM span carries the staged
        ``metric_collection`` and ``metadata``; agent span carries
        nothing extra (no agent-span staging)."""
        agent = create_next_span_agent(
            name="pydanticai-next-llm-only-test",
            tags=["pydanticai", "next-llm"],
            metadata={"test_type": "next_llm_only"},
            thread_id="next-llm-only-123",
            user_id="test-user",
        )

        result = invoke_with_next_llm_span(
            "Say hello in exactly three words.",
            agent=agent,
            llm_metric_collection="llm_metrics_only_v1",
            llm_metadata={
                "prompt_variant": "B",
                "purpose": "next_llm_only",
            },
        )

        assert result is not None
        assert len(result) > 0

    @trace_test("pydanticai_next_stacked_schema.json")
    def test_next_stacked_agent_and_llm(self):
        """``with next_agent_span(...), next_llm_span(...)`` stacked:
        agent span gets agent-staged values, LLM span gets LLM-staged
        values, no cross-talk between typed slots."""
        agent = create_next_span_agent(
            name="pydanticai-next-stacked-test",
            tags=["pydanticai", "stacked"],
            metadata={"test_type": "next_stacked"},
            thread_id="next-stacked-123",
            user_id="test-user",
        )

        result = invoke_with_stacked_next_spans(
            "Say goodbye in exactly three words.",
            agent=agent,
            agent_metric_collection="agent_stacked_v1",
            llm_metric_collection="llm_stacked_v1",
            agent_metadata={"layer": "agent", "scenario": "stacked"},
            llm_metadata={"layer": "llm", "scenario": "stacked"},
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# EXECUTION MODES TESTS (Mode 2: with trace, Mode 3: @observe,
#                       Mode 1 + tool-driven trace enrichment)
# =============================================================================


class TestExecutionModes:
    """Schema-asserted coverage for the three execution modes documented
    in ``deepeval/integrations/pydantic_ai/README.md``. The other
    schema tests in this file all run in Mode 1 (bare ``agent.run``);
    these add Mode 2 / Mode 3 / Mode-1-with-tool-enrichment."""

    @trace_test("pydanticai_observe_mode_schema.json")
    def test_observe_mode(self):
        """Mode 3 — ``@observe(type="agent")`` wraps the agent call.
        Trace routing flips to REST via the user-pushed (non-implicit)
        trace context; the captured trace tree shows the deepeval-managed
        outer agent span containing pydantic-ai's own agent/llm spans."""
        agent = create_modes_agent(
            name="pydanticai-observe-mode-test",
            tags=["pydanticai", "observe-mode"],
            metadata={"test_type": "observe_mode"},
            thread_id="observe-mode-123",
            user_id="test-user",
        )

        result = invoke_in_observe_mode(
            "Say hello in exactly three words.",
            agent=agent,
            outer_name="observe_outer",
            trace_name="pydanticai-observe-trace",
            user_id="observe-user",
            tags=["observe-mode", "runtime"],
            metadata={"mode": "observe", "source": "runtime"},
        )

        assert result is not None
        assert len(result) > 0

    @trace_test("pydanticai_with_trace_mode_schema.json")
    def test_with_trace_mode(self):
        """Mode 2 — ``with trace(...)`` wraps the agent call. Like Mode 3
        for routing, but no outer deepeval-managed span — the captured
        tree is just pydantic-ai's spans under the user-pushed trace."""
        agent = create_modes_agent(
            name="pydanticai-with-trace-mode-test",
            tags=["pydanticai", "with-trace"],
            metadata={"test_type": "with_trace_mode"},
            thread_id="with-trace-mode-123",
            user_id="test-user",
        )

        result = invoke_in_with_trace_mode(
            "Say goodbye in exactly three words.",
            agent=agent,
            trace_name="pydanticai-with-trace",
            user_id="with-trace-user",
            thread_id="with-trace-thread",
            tags=["with-trace", "runtime"],
            metadata={"mode": "with_trace", "source": "runtime"},
        )

        assert result is not None
        assert len(result) > 0

    @trace_test("pydanticai_bare_tool_enrichment_schema.json")
    def test_bare_trace_enrichment_from_tool(self):
        """Mode 1 + ``update_current_trace`` from inside a tool body.
        No ``@observe`` / ``with trace(...)``: the implicit ``Trace``
        placeholder pushed by ``SpanInterceptor`` is the write target.
        Mirrors ``pydantic_after_bare.py``."""
        agent = create_enrichment_agent(
            name="pydanticai-bare-enrichment-test",
            tags=["pydanticai", "enrichment"],
            metadata={"test_type": "bare_tool_enrichment"},
            thread_id="bare-enrichment-123",
            user_id="test-user",
        )

        result = invoke_with_tool_enrichment(
            "Use the lookup tool with key 'foobar' and report the result.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# THREAD ISOLATION (behavioral, NO schema)
# =============================================================================


class TestThreadIsolation:
    """Behavioral isolation check across a ``ThreadPoolExecutor``.

    Mirrors ``pydantic_after_threads.py``. **No ``@trace_test``
    decorator** — ``trace_testing_manager.test_dict`` is a single
    global slot and would race across the 3 concurrent
    ``end_trace`` calls, capturing only the (random) last winner.
    The interesting property here is contextvar isolation in user
    space, which we can assert without touching the trace capture.
    """

    def test_thread_isolation(self):
        """Three concurrent ``agent.run_sync`` calls from different
        worker threads. Each worker stamps ``_request_ctx`` with its
        own ``(user_id, request_id)`` before the call and re-reads it
        after. The post-run value MUST equal the pre-run value
        (no cross-thread leakage of ``ContextVar`` state, no
        leakage through pydantic-ai's anyio thread bridge to the
        sync tool body, no leakage through deepeval's
        ``current_trace_context`` / ``current_span_context``
        contextvars).
        """
        agent = create_isolation_agent(name="pydanticai-thread-isolation-test")
        requests = make_distinct_requests()

        results = threaded_isolation_run(agent, requests)

        # All three calls returned a result.
        assert len(results) == len(requests)

        # Per-task contextvar stability: post-run value matches pre-run.
        # If this fails, either ContextVar was leaking across threads or
        # pydantic-ai's anyio bridge didn't carry the context into the
        # tool body (and the tool's no-op write back into the ctx wouldn't
        # be visible — but we only ``set`` in the worker, never the tool).
        for r in results:
            assert r["post_run_request_id"] == r["request_id"], (
                f"Thread {r.get('thread_name')!r} saw request_id "
                f"{r['post_run_request_id']!r} after agent.run, "
                f"expected {r['request_id']!r}. ContextVar leak across "
                "threads."
            )
            assert r["post_run_user_id"] == r["user_id"], (
                f"Thread {r.get('thread_name')!r} saw user_id "
                f"{r['post_run_user_id']!r} after agent.run, "
                f"expected {r['user_id']!r}."
            )

        # All request_ids and user_ids are distinct across threads
        # (sanity guard — if these collapse to one value, the
        # ``ContextVar.set`` in one worker stomped another's).
        assert len({r["request_id"] for r in results}) == len(requests)
        assert len({r["user_id"] for r in results}) == len(requests)

        # Each worker's output reflects its own ``key`` (the LLM was
        # told to call ``get_data`` with that key, and the tool returns
        # ``data-for-<key>``). If outputs got mixed across threads,
        # this fails.
        for r in results:
            assert r["expected_key"] in r["output"], (
                f"Thread {r.get('thread_name')!r} expected output to "
                f"reference key {r['expected_key']!r}, got "
                f"{r['output']!r}. Possible cross-thread output mix."
            )
