"""
Async OpenAI Agents Tests
All asynchronous tests using Runner.run() and Runner.run_streamed()
"""

import os
import pytest
from deepeval.openai_agents.runner import Runner
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
from tests.test_integrations.test_openai_agents.apps.streaming_app import get_streaming_app

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

class TestSimpleAppAsync:
    """Tests for simple single-agent async execution."""

    @trace_test("openai_simple_async.json")
    async def test_simple_async(self):
        agent, user_input = get_simple_app()
        await Runner.run(agent, user_input)

# =============================================================================
# TOOL CHAIN TESTS
# =============================================================================

class TestToolChainAppAsync:
    """Tests for async sequential tool dependencies."""

    @trace_test("openai_tool_chain_async.json")
    async def test_tool_chain_async(self):
        agent, user_input = get_tool_chain_app()
        await Runner.run(agent, user_input)

# =============================================================================
# HANDOFF TESTS
# =============================================================================

class TestHandoffAppAsync:
    """Tests for async multi-agent handoffs."""

    @trace_test("openai_handoff_chain_async.json")
    async def test_handoff_chain_async(self):
        agent, user_input = get_handoff_chain_app()
        await Runner.run(agent, user_input)

# =============================================================================
# CONTEXT TESTS
# =============================================================================

class TestContextAppAsync:
    """Tests for async thread_id and user_id injection."""

    @trace_test("openai_thread_context_async.json")
    async def test_thread_context_async(self):
        agent, user_input = get_thread_context_app()
        await Runner.run(
            agent, 
            user_input, 
            thread_id="thread_async_789", 
            user_id="user_async_000"
        )

# =============================================================================
# STREAMING TESTS
# =============================================================================

class TestStreamingAppAsync:
    """Tests for streaming execution."""

    @trace_test("openai_streaming_async.json")
    async def test_streaming_async(self):
        agent, user_input = get_streaming_app()
        
        # Execute streaming run
        result_stream = Runner.run_streamed(agent, user_input)
        
        # Consume the stream to trigger all events and close spans
        async for _ in result_stream.stream_events():
            pass