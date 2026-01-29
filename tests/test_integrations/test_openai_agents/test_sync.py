"""
Sync OpenAI Agents Tests
All synchronous tests using Runner.run_sync()

NOTE: Run with GENERATE_SCHEMAS=1 first to generate the JSON schemas.
"""

import os
import pytest
from agents import Runner, trace

from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

# App imports
from tests.test_integrations.test_openai_agents.apps.simple_agent import (
    agent as simple_agent,
)
from tests.test_integrations.test_openai_agents.apps.tool_agent import (
    agent as tool_agent,
)
from tests.test_integrations.test_openai_agents.apps.eval_agent import (
    agent as eval_agent,
)
from tests.test_integrations.test_openai_agents.apps.handoff_agent import (
    triage_agent,
)
from tests.test_integrations.test_openai_agents.apps.session_agent import (
    get_agent as get_session_agent,
    get_session,
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
        # Ensure directory exists
        os.makedirs(_schemas_dir, exist_ok=True)
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)


class TestSimpleAgent:
    @trace_test("openai_agents_simple_schema.json")
    def test_simple_greeting(self):
        """Test a simple greeting with standard Agent."""
        with trace(
            workflow_name="openai_agents_simple",
            metadata={"test_type": "simple", "tags": ["openai_agents", "simple"]},
        ):
            result = Runner.run_sync(simple_agent, "Hello")
            assert result.final_output


class TestToolAgent:
    @trace_test("openai_agents_tool_weather_schema.json")
    def test_tool_weather(self):
        """Test weather tool with DeepEval wrapper."""
        with trace(
            workflow_name="openai_agents_tool_weather",
            metadata={"tags": ["openai_agents", "tool"]},
        ):
            result = Runner.run_sync(tool_agent, "Weather in London")
            # Case insensitive check
            assert "rainy" in result.final_output.lower()

    @trace_test("openai_agents_tool_math_schema.json")
    def test_tool_calculation(self):
        """Test calculation tool."""
        with trace(
            workflow_name="openai_agents_tool_math",
            metadata={"tags": ["openai_agents", "tool", "math"]},
        ):
            result = Runner.run_sync(tool_agent, "Calculate 10 + 5")
            assert "15" in result.final_output


class TestEvalAgent:
    @trace_test("openai_agents_eval_schema.json")
    def test_eval_agent_metrics(self):
        """Test DeepEvalAgent with metric collections."""
        with trace(
            workflow_name="openai_agents_eval",
            metadata={"tags": ["openai_agents", "eval"]},
        ):
            result = Runner.run_sync(eval_agent, "Say hi")
            assert result.final_output


class TestHandoffAgent:
    @trace_test("openai_agents_handoff_spanish_schema.json")
    def test_handoff_spanish(self):
        """Test handoff to Spanish agent."""
        with trace(
            workflow_name="openai_agents_handoff",
            metadata={"tags": ["openai_agents", "handoff"]},
        ):
            result = Runner.run_sync(triage_agent, "Hola")
            assert "Hola" in result.final_output


class TestSessionAgent:
    @trace_test("openai_agents_session_schema.json")
    def test_session_memory(self):
        """Test memory across turns."""
        agent = get_session_agent()
        session = get_session("sync_sess_1")

        with trace(workflow_name="openai_agents_session", group_id="sync_sess_1"):
            Runner.run_sync(agent, "My name is Bob", session=session)

        with trace(workflow_name="openai_agents_session", group_id="sync_sess_1"):
            result = Runner.run_sync(agent, "What is my name?", session=session)
            assert "Bob" in result.final_output