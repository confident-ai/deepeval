"""
tests/test_integrations/test_openai_agents/test_sync.py
Sync OpenAI Agents Tests
"""

import os
import pytest
from agents import Runner, trace, add_trace_processor, SQLiteSession
from deepeval.openai_agents import DeepEvalTracingProcessor

from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)

# App imports
from tests.test_integrations.test_openai_agents.apps.simple_app import (
    get_simple_app,
)
from tests.test_integrations.test_openai_agents.apps.tool_chain_app import (
    get_tool_chain_app,
)
from tests.test_integrations.test_openai_agents.apps.handoff_chain_app import (
    get_handoff_chain_app,
)
from tests.test_integrations.test_openai_agents.apps.thread_context_app import (
    get_thread_context_app,
)
from tests.test_integrations.test_openai_agents.apps.evals_app import (
    get_evals_app,
)
from tests.test_integrations.test_openai_agents.apps.structured_output_app import (
    get_structured_output_app,
)
from tests.test_integrations.test_openai_agents.apps.session_app import (
    get_session_app,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

add_trace_processor(DeepEvalTracingProcessor())


GENERATE_MODE = os.environ.get("GENERATE_SCHEMAS", "").lower() in (
    "true",
    "1",
    "yes",
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    """
    Decorator that switches between generate and assert mode based on GENERATE_MODE.
    """
    schema_path = os.path.join(_schemas_dir, schema_name)
    if GENERATE_MODE:
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


class TestOpenAIAgentsSync:
    """Tests for synchronous OpenAI Agents execution."""

    @trace_test("openai_simple_sync.json")
    def test_simple_sync(self):
        """Test basic single-agent execution."""
        agent, user_input = get_simple_app()
        Runner.run_sync(agent, user_input)

    @trace_test("openai_tool_chain_sync.json")
    def test_tool_chain_sync(self):
        """Test sequential tool dependencies (Reasoning Loop)."""
        agent, user_input = get_tool_chain_app()
        Runner.run_sync(agent, user_input)

    @trace_test("openai_handoff_chain_sync.json")
    def test_handoff_chain_sync(self):
        """Test multi-agent handoffs (Trace continuity)."""
        agent, user_input = get_handoff_chain_app()
        Runner.run_sync(agent, user_input)

    @trace_test("openai_thread_context_sync.json")
    def test_thread_context_sync(self):
        """Test thread_id and metadata injection via trace context."""
        agent, user_input = get_thread_context_app()
        with trace(
            workflow_name="trace_sync",
            group_id="thread_sync_123",
            metadata={"test_run": "sync_context"},
        ):
            Runner.run_sync(agent, user_input)

    @trace_test("openai_evals_sync.json")
    def test_evals_sync(self):
        """Test metric collections and prompt logging."""
        agent, user_input = get_evals_app()
        Runner.run_sync(agent, user_input)

    @trace_test("openai_structured_sync.json")
    def test_structured_output_sync(self):
        """Test agents with Pydantic output_type."""
        agent, user_input = get_structured_output_app()
        Runner.run_sync(agent, user_input)

    @trace_test("openai_session_sync.json")
    def test_session_sync(self):
        """Test multi-turn conversations with SQLiteSession."""
        agent, session, inputs = get_session_app("session_sync_1")

        if session is None:
            pytest.skip("SQLiteSession not available")

        # Turn 1: Set Memory
        Runner.run_sync(agent, inputs[0], session=session)

        # Turn 2: Retrieve Memory (Traced)
        Runner.run_sync(agent, inputs[1], session=session)
