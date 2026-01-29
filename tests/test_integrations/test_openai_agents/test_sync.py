"""
Sync OpenAI Agents Tests
All synchronous tests using Runner.run_sync()
"""

import os
from agents import Runner
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

# App imports
from tests.test_integrations.test_openai_agents.apps.simple_app import get_simple_app
from tests.test_integrations.test_openai_agents.apps.tool_chain_app import get_tool_chain_app
from tests.test_integrations.test_openai_agents.apps.handoff_chain_app import get_handoff_chain_app
from tests.test_integrations.test_openai_agents.apps.thread_context_app import get_thread_context_app

# =============================================================================
# CONFIGURATION
# =============================================================================

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")

def trace_test(schema_name: str):
    """Decorator for schema generation/assertion."""
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)

# =============================================================================
# SIMPLE AGENT TESTS
# =============================================================================

class TestSimpleApp:
    """Tests for simple single-agent execution."""

    @trace_test("openai_simple_sync.json")
    def test_simple_sync(self):
        agent, user_input = get_simple_app()
        Runner.run_sync(agent, user_input)

# =============================================================================
# TOOL CHAIN TESTS
# =============================================================================

class TestToolChainApp:
    """Tests for sequential tool dependencies."""

    @trace_test("openai_tool_chain_sync.json")
    def test_tool_chain_sync(self):
        agent, user_input = get_tool_chain_app()
        # Force the tool execution loop
        Runner.run_sync(agent, user_input)

# =============================================================================
# HANDOFF TESTS
# =============================================================================

class TestHandoffApp:
    """Tests for multi-agent handoffs."""

    @trace_test("openai_handoff_chain_sync.json")
    def test_handoff_chain_sync(self):
        agent, user_input = get_handoff_chain_app()
        Runner.run_sync(agent, user_input)

# =============================================================================
# CONTEXT TESTS
# =============================================================================

class TestContextApp:
    """Tests for thread_id and user_id injection."""

    @trace_test("openai_thread_context_sync.json")
    def test_thread_context_sync(self):
        agent, user_input = get_thread_context_app()
        Runner.run_sync(
            agent, 
            user_input, 
            thread_id="thread_sync_123", 
            user_id="user_sync_456"
        )