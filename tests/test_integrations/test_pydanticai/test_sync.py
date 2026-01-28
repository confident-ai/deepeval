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

# App imports
from tests.test_integrations.test_pydanticai.apps.pydanticai_simple_app import (
    create_simple_agent,
    invoke_simple_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_tool_app import (
    create_tool_agent,
    invoke_tool_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_prompt_app import (
    create_prompt_agent,
    invoke_prompt_agent,
)
from tests.test_integrations.test_pydanticai.apps.pydanticai_metric_collection_app import (
    create_trace_metric_collection_agent,
    create_agent_metric_collection_agent,
    create_llm_metric_collection_agent,
    invoke_metric_collection_agent,
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

    @trace_test("pydanticai_tool_metric_collection_schema.json")
    def test_tool_metric_collection(self):
        """Test tool metric collection mapping."""
        agent = create_tool_agent(
            name="pydanticai-tool-metric-test",
            tags=["pydanticai", "tool", "metric-collection"],
            metadata={"test_type": "tool_metric_collection"},
            thread_id="tool-metric-123",
            user_id="test-user",
            tool_metric_collection_map={"calculate": "calculator-metrics"},
        )

        result = invoke_tool_agent(
            "What is 15 plus 25?",
            agent=agent,
        )

        assert result is not None
        assert "40" in result


# =============================================================================
# PROMPT TESTS (Confident Prompt attribution)
# =============================================================================


class TestPromptApp:
    """Tests for PydanticAI agent with Confident Prompt logging."""

    @trace_test("pydanticai_prompt_schema.json")
    def test_prompt_attribution(self):
        """Test that confident_prompt is logged in trace attributes."""
        agent = create_prompt_agent(
            prompt_alias="test-prompt-alias",
            prompt_version="01.00.00",
            name="pydanticai-prompt-test",
            tags=["pydanticai", "prompt"],
            metadata={"test_type": "prompt"},
            thread_id="prompt-123",
            user_id="test-user",
        )

        result = invoke_prompt_agent(
            "Say hello in exactly two words.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# METRIC COLLECTION TESTS (Online evals)
# =============================================================================


class TestMetricCollectionApp:
    """Tests for PydanticAI agent with metric collection settings."""

    @trace_test("pydanticai_trace_metric_collection_schema.json")
    def test_trace_metric_collection(self):
        """Test trace-level metric collection attribute."""
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

    @trace_test("pydanticai_agent_metric_collection_schema.json")
    def test_agent_metric_collection(self):
        """Test agent-span-level metric collection attribute."""
        agent = create_agent_metric_collection_agent(
            metric_collection="test-agent-metrics",
            name="pydanticai-agent-metric-test",
            tags=["pydanticai", "agent-metric-collection"],
            metadata={"test_type": "agent_metric_collection"},
            thread_id="agent-metric-123",
            user_id="test-user",
        )

        result = invoke_metric_collection_agent(
            "Say hello in exactly two words.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0

    @trace_test("pydanticai_llm_metric_collection_schema.json")
    def test_llm_metric_collection(self):
        """Test LLM-span-level metric collection attribute."""
        agent = create_llm_metric_collection_agent(
            metric_collection="test-llm-metrics",
            name="pydanticai-llm-metric-test",
            tags=["pydanticai", "llm-metric-collection"],
            metadata={"test_type": "llm_metric_collection"},
            thread_id="llm-metric-123",
            user_id="test-user",
        )

        result = invoke_metric_collection_agent(
            "Say hello in exactly two words.",
            agent=agent,
        )

        assert result is not None
        assert len(result) > 0
