"""
tests/test_integrations/test_openai_agents/test_async.py
Async OpenAI Agents Tests
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
from tests.test_integrations.test_openai_agents.apps.streaming_app import (
    get_streaming_app,
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


class TestOpenAIAgentsAsync:
    """Tests for asynchronous OpenAI Agents execution."""

    @trace_test("openai_simple_async.json")
    async def test_simple_async(self):
        """Test basic single-agent async execution."""
        agent, user_input = get_simple_app()
        await Runner.run(agent, user_input)

    @trace_test("openai_tool_chain_async.json")
    async def test_tool_chain_async(self):
        """Test async sequential tool dependencies."""
        agent, user_input = get_tool_chain_app()
        await Runner.run(agent, user_input)

    @trace_test("openai_handoff_chain_async.json")
    async def test_handoff_chain_async(self):
        """Test async multi-agent handoffs."""
        agent, user_input = get_handoff_chain_app()
        await Runner.run(agent, user_input)

    @trace_test("openai_thread_context_async.json")
    async def test_thread_context_async(self):
        """Test async context propagation."""
        agent, user_input = get_thread_context_app()
        with trace(
            workflow_name="trace_async",
            group_id="thread_async_789",
            metadata={"test_run": "async_context"},
        ):
            await Runner.run(agent, user_input)

    @trace_test("openai_evals_async.json")
    async def test_evals_async(self):
        """Test async evals and prompt logging."""
        agent, user_input = get_evals_app()
        await Runner.run(agent, user_input)

    @trace_test("openai_structured_async.json")
    async def test_structured_output_async(self):
        """Test async agents with Pydantic output_type."""
        agent, user_input = get_structured_output_app()
        await Runner.run(agent, user_input)

    @trace_test("openai_session_async.json")
    async def test_session_async(self):
        """Test async multi-turn conversations with SQLiteSession."""
        agent, session, inputs = get_session_app("session_async_1")

        if session is None:
            pytest.skip("SQLiteSession not available")

        # Turn 1: Set Memory
        await Runner.run(agent, inputs[0], session=session)

        # Turn 2: Retrieve Memory (Traced)
        await Runner.run(agent, inputs[1], session=session)

    @trace_test("openai_streaming_async.json")
    async def test_streaming_async(self):
        """Test streaming execution."""
        agent, user_input = get_streaming_app()
        result_stream = Runner.run_streamed(agent, user_input)

        async for _ in result_stream.stream_events():
            pass
