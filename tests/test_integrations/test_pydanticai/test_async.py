"""
Async PydanticAI Tests
All asynchronous tests using deterministic settings.
"""

import os
import pytest
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)
from tests.test_integrations.test_pydanticai.apps.eval_app import (
    create_evals_agent,
    ainvoke_evals_agent,
)

# App imports
from tests.test_integrations.test_pydanticai.apps.pydanticai_simple_app import (
    create_simple_agent,
    ainvoke_simple_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_tool_app import (
    create_tool_agent,
    ainvoke_tool_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_streaming_app import (
    create_streaming_agent,
    stream_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_multiple_tools_app import (
    create_multiple_tools_agent,
    ainvoke_multiple_tools_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_isolation_app import (
    concurrent_isolation_run,
    create_isolation_agent,
    make_distinct_requests,
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
# ASYNC SIMPLE APP TESTS (LLM only, no tools)
# =============================================================================


class TestAsyncSimpleApp:
    """Async tests for simple LLM-only PydanticAI agent."""

    @pytest.mark.asyncio
    @trace_test("pydanticai_async_simple_schema.json")
    async def test_async_simple_greeting(self):
        """Test a simple async greeting that returns a response."""
        agent = create_simple_agent(
            name="pydanticai-async-simple-test",
            tags=["pydanticai", "simple", "async"],
            metadata={"test_type": "async_simple"},
            thread_id="async-simple-123",
            user_id="test-user-async",
        )

        result = await ainvoke_simple_agent(
            "Say goodbye in exactly three words.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# ASYNC MULTIPLE TOOLS TESTS
# =============================================================================


class TestAsyncMultipleToolsApp:
    """Async tests for PydanticAI agent with multiple tools."""

    @pytest.mark.asyncio
    @trace_test("pydanticai_async_parallel_tools_schema.json")
    async def test_async_parallel_tool_calls(self):
        """Test async parallel tool calls with both get_weather and get_time."""
        agent = create_multiple_tools_agent(
            name="pydanticai-async-parallel-tools",
            tags=["pydanticai", "parallel-tools", "async"],
            metadata={"test_type": "async_parallel_tools"},
            thread_id="async-parallel-tools-123",
            user_id="test-user-async",
        )

        result = await ainvoke_multiple_tools_agent(
            "Use both the get_weather tool AND the get_time tool for Tokyo. "
            "Call both tools exactly once each.",
            agent=agent,
        )

        assert result is not None
        # Verify both weather and time data are in response
        # Weather: Tokyo is "Sunny, 72F"
        assert "72" in result or "sunny" in result.lower()
        # Time: Tokyo is "3:00 PM JST"
        assert "3:00" in result or "JST" in result


# =============================================================================
# ASYNC TOOL APP TESTS (Agent with tool calling)
# =============================================================================


class TestAsyncToolApp:
    """Async tests for PydanticAI agent with tool calling."""

    @pytest.mark.asyncio
    @trace_test("pydanticai_async_tool_schema.json")
    async def test_async_tool_calculation(self):
        """Test an async calculation using a tool."""
        agent = create_tool_agent(
            name="pydanticai-async-tool-test",
            tags=["pydanticai", "tool", "async"],
            metadata={"test_type": "async_tool"},
            thread_id="async-tool-123",
            user_id="test-user-async",
        )

        result = await ainvoke_tool_agent(
            "What is 9 multiplied by 6?",
            agent=agent,
        )

        assert result is not None
        assert "54" in result


# =============================================================================
# STREAMING TESTS
# =============================================================================


class TestStreamingApp:
    """Tests for PydanticAI agent with streaming response."""

    @pytest.mark.asyncio
    @trace_test("pydanticai_streaming_schema.json")
    async def test_streaming_response(self):
        """Test streaming response collection."""
        agent = create_streaming_agent(
            name="pydanticai-streaming-test",
            tags=["pydanticai", "streaming"],
            metadata={"test_type": "streaming"},
            thread_id="streaming-123",
            user_id="test-user-streaming",
        )

        result = await stream_agent(
            "Say hello in exactly three words.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# DEEPEVAL FEATURES TESTS
# =============================================================================


class TestDeepEvalFeaturesAsync:
    """Async tests for DeepEval-specific trace-level settings + metadata."""

    @pytest.mark.asyncio
    @trace_test("pydanticai_features_async.json")
    async def test_full_features_async(self):
        """Async equivalent of ``test_full_features_sync``."""
        agent = create_evals_agent(
            metric_collection="trace_metrics_override_async_v1",
            name="pydanticai-full-features-async",
            tags=["pydanticai", "features", "async"],
            metadata={"env": "testing_async", "mode": "async"},
            thread_id="thread-async-features-002",
            user_id="user-async-002",
        )

        result = await ainvoke_evals_agent(
            "Use the special_tool to process 'Async Data'",
            agent=agent,
            agent_metric_collection="agent_metrics_async_v1",
        )

        assert result is not None


# =============================================================================
# CONCURRENT ISOLATION (behavioral, NO schema)
# =============================================================================


class TestConcurrentIsolation:
    """Behavioral isolation check across ``asyncio.gather``.

    Mirrors ``pydantic_after_concurrent.py``. **No ``@trace_test``
    decorator** — ``trace_testing_manager.test_dict`` is a single
    global slot that would race across the 3 concurrent
    ``end_trace`` calls, capturing only the (random) last winner.
    The interesting property here is contextvar isolation in user
    space, which we can assert without touching the trace capture.
    """

    @pytest.mark.asyncio
    async def test_concurrent_isolation(self):
        """Three concurrent ``await agent.run(...)`` calls via
        ``asyncio.gather``. Each task stamps ``_request_ctx`` with
        its own ``(user_id, request_id)`` before the call and re-reads
        it after. The post-run value MUST equal the pre-run value
        (no cross-task leakage of ``ContextVar`` state, no leakage
        through deepeval's ``current_trace_context`` /
        ``current_span_context`` per-task copies).
        """
        agent = create_isolation_agent(
            name="pydanticai-concurrent-isolation-test"
        )
        requests = make_distinct_requests()

        results = await concurrent_isolation_run(agent, requests)

        # All three calls returned a result.
        assert len(results) == len(requests)

        # Per-task contextvar stability: post-run value matches pre-run.
        # If this fails, ContextVar state leaked across asyncio tasks
        # (which would be a serious regression — Python guarantees task-
        # local contextvars via per-task context snapshots).
        for r in results:
            assert r["post_run_request_id"] == r["request_id"], (
                f"Task for request_id={r['request_id']!r} saw "
                f"post-run value {r['post_run_request_id']!r}. "
                "ContextVar leak across asyncio tasks."
            )
            assert r["post_run_user_id"] == r["user_id"], (
                f"Task for user_id={r['user_id']!r} saw post-run "
                f"value {r['post_run_user_id']!r}."
            )

        # All request_ids and user_ids are distinct across tasks.
        assert len({r["request_id"] for r in results}) == len(requests)
        assert len({r["user_id"] for r in results}) == len(requests)

        # Each task's output reflects its own ``key``.
        for r in results:
            assert r["expected_key"] in r["output"], (
                f"Task expected output to reference key "
                f"{r['expected_key']!r}, got {r['output']!r}. "
                "Possible cross-task output mix."
            )
