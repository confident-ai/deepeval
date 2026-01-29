"""
Async OpenAI Agents Tests
All asynchronous tests using Runner.run()
"""

import os
import pytest
from agents import Runner, trace

from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

from tests.test_integrations.test_openai_agents.apps.simple_agent import (
    agent as simple_agent,
)
from tests.test_integrations.test_openai_agents.apps.tool_agent import (
    agent as tool_agent,
)
from tests.test_integrations.test_openai_agents.apps.handoff_agent import (
    triage_agent,
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
        os.makedirs(_schemas_dir, exist_ok=True)
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


class TestAsyncSimpleAgent:
    @pytest.mark.asyncio
    @trace_test("openai_agents_async_simple_schema.json")
    async def test_async_greeting(self):
        with trace(
            workflow_name="openai_agents_async_simple",
            metadata={"tags": ["openai_agents", "async"]},
        ):
            result = await Runner.run(simple_agent, "Hello")
            assert result.final_output


class TestAsyncToolAgent:
    @pytest.mark.asyncio
    @trace_test("openai_agents_async_tool_schema.json")
    async def test_async_tool(self):
        with trace(
            workflow_name="openai_agents_async_tool",
            metadata={"tags": ["openai_agents", "async", "tool"]},
        ):
            result = await Runner.run(tool_agent, "Weather in Tokyo")
            assert "cloudy" in result.final_output.lower()


class TestAsyncHandoffAgent:
    @pytest.mark.asyncio
    @trace_test("openai_agents_async_handoff_schema.json")
    async def test_async_handoff(self):
        with trace(
            workflow_name="openai_agents_async_handoff",
            metadata={"tags": ["openai_agents", "async", "handoff"]},
        ):
            result = await Runner.run(triage_agent, "Hello")
            assert "Hello" in result.final_output
