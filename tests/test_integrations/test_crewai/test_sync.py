"""
tests/test_integrations/test_crewai/test_sync.py
Sync CrewAI Tests
"""

import os
import pytest
from deepeval.integrations.crewai import instrument_crewai
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)

# App imports
from tests.test_integrations.test_crewai.apps.simple_app import get_simple_app
from tests.test_integrations.test_crewai.apps.multi_agent_app import get_multi_agent_app
from tests.test_integrations.test_crewai.apps.tool_usage_app import get_tool_usage_app
from tests.test_integrations.test_crewai.apps.knowledge_retriever_app import get_knowledge_app
from tests.test_integrations.test_crewai.apps.hierarchical_app import get_hierarchical_app

# =============================================================================
# CONFIGURATION
# =============================================================================

instrument_crewai()

# Set to True to generate schemas, False to assert against existing schemas
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


class TestCrewAISync:
    """Tests for synchronous CrewAI execution."""

    @trace_test("crewai_simple_kickoff.json")
    def test_simple_kickoff(self):
        """Test basic single-agent kickoff."""
        crew = get_simple_app(id_suffix="_sync")
        result = crew.kickoff(inputs={"input": "Hello World"})
        assert result is not None

    @trace_test("crewai_multi_agent_sequential.json")
    def test_multi_agent_flow(self):
        """Test sequential multi-agent flow (Researcher -> Writer)."""
        crew = get_multi_agent_app()
        # No inputs needed as tasks are hardcoded for this demo
        result = crew.kickoff()
        assert result is not None

    @trace_test("crewai_tool_usage.json")
    def test_tool_usage(self):
        """Test capture of tool inputs and outputs."""
        crew = get_tool_usage_app()
        result = crew.kickoff(inputs={"city": "Paris"})
        assert "Weather" in str(result)

    @trace_test("crewai_knowledge_retrieval.json")
    def test_knowledge_retrieval(self):
        """Test capture of KnowledgeRetrieval events."""
        crew = get_knowledge_app()
        result = crew.kickoff()
        assert result is not None

    @trace_test("crewai_hierarchical.json")
    def test_hierarchical_process(self):
        """Test hierarchical process with manager delegation."""
        # Note: This requires an OpenAI API key or mock in the environment
        # for the manager LLM to function correctly.
        crew = get_hierarchical_app()
        result = crew.kickoff()
        assert result is not None

    @trace_test("crewai_kickoff_for_each.json")
    def test_kickoff_for_each(self):
        """Test running the same task for multiple inputs synchronously."""
        crew = get_simple_app(id_suffix="_foreach")
        inputs = [{"input": "User A"}, {"input": "User B"}]
        results = crew.kickoff_for_each(inputs=inputs)
        assert len(results) == 2