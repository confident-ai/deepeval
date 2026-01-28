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
