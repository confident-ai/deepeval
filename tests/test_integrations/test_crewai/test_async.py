"""
tests/test_integrations/test_crewai/test_async.py
Async CrewAI Tests
"""

import os
import pytest
from deepeval.integrations.crewai import instrument_crewai, reset_crewai_instrumentation
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.trace_test_manager import trace_testing_manager

# App imports
from tests.test_integrations.test_crewai.apps.simple_app import get_simple_app
from tests.test_integrations.test_crewai.apps.async_app import get_async_app
from tests.test_integrations.test_crewai.apps.tool_usage_app import (
    get_tool_usage_app,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

instrument_crewai()

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


class TestCrewAIAsync:
    """Tests for asynchronous CrewAI execution."""

    @pytest.mark.asyncio
    @trace_test("crewai_async_kickoff.json")
    async def test_async_kickoff(self):
        """Test basic async kickoff."""
        crew = get_async_app()
        result = await crew.kickoff_async(inputs={"input": "Async Request 1"})
        assert result is not None

    @pytest.mark.asyncio
    @trace_test("crewai_async_tool_usage.json")
    async def test_async_tool_usage(self):
        """Test async kickoff with tool usage."""
        crew = get_tool_usage_app()
        result = await crew.kickoff_async(inputs={"city": "Tokyo"})
        assert "Weather" in str(result)

    @pytest.mark.asyncio
    @trace_test("crewai_kickoff_for_each_async.json")
    async def test_kickoff_for_each_async(self):
        """Test async batch processing (kickoff_for_each_async)."""
        crew = get_simple_app(id_suffix="_async_batch")
        inputs = [{"input": "Batch 1"}, {"input": "Batch 2"}]
        results = await crew.kickoff_for_each_async(inputs=inputs)
        assert len(results) == 2

    @pytest.mark.asyncio
    @trace_test("crewai_akickoff.json")
    async def test_akickoff_alias(self):
        """
        Test the 'akickoff' alias (present in newer CrewAI versions).
        """
        crew = get_simple_app(id_suffix="_akickoff")

        # Guard clause for older CrewAI versions
        if not hasattr(crew, "akickoff"):
            pytest.skip("akickoff method not found on Crew object")

        result = await crew.akickoff(inputs={"input": "Testing Alias"})
        assert result is not None

    @pytest.fixture(autouse=True)
    def reset_instrumentation(self):
        """Reset ALL tracing state before each test."""
        reset_crewai_instrumentation()
        trace_manager.clear_traces()
        test_exporter.clear_span_json_list()
        trace_testing_manager.test_dict = None
        yield
