"""
tests/test_integrations/test_crewai/test_stress.py
Stress/Concurrency tests for CrewAI Integration
"""

import pytest
import asyncio
from deepeval.integrations.crewai import instrument_crewai

# App imports
from tests.test_integrations.test_crewai.apps.simple_app import get_simple_app

instrument_crewai()


@pytest.mark.asyncio
async def test_concurrent_crews_isolation():
    """
    Verify that running two crews concurrently (e.g., handling two different user requests)
    does not cause the instrumentation to crash or mix up contexts.

    This is a regression test for "Span mismatch" errors that can occur when
    global event listeners aren't thread/task-aware.
    """

    # Create two distinct crews
    crew1 = get_simple_app(id_suffix="_stress_1")
    crew2 = get_simple_app(id_suffix="_stress_2")

    async def run_crew_1():
        return await crew1.kickoff_async(inputs={"input": "User 1 Request"})

    async def run_crew_2():
        return await crew2.kickoff_async(inputs={"input": "User 2 Request"})

    # Run them concurrently in the same event loop
    results = await asyncio.gather(run_crew_1(), run_crew_2())

    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None

    # Basic check to ensure no spans were left dangling in the active manager
    # (Note: This assumes the trace manager cleans up correctly after a successful run)
    # If traces are being sent to the API background thread, active_spans might not be immediately empty,
    # but the context vars should be clear.

    from deepeval.tracing.context import (
        current_span_context,
        current_trace_context,
    )

    assert current_span_context.get() is None
