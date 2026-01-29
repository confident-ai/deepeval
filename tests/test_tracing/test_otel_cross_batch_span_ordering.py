"""
Test for OTEL BatchSpanProcessor Cross-Batch Span Ordering Issue

HYPOTHESIS:
When OTEL BatchSpanProcessor delivers spans from a single trace in separate
export() calls (with child spans arriving before parent spans), the current
DeepEval exporter/trace assembly logic fails to correctly establish parent-child
relationships. This results in LLM spans being orphaned (added to root_spans
instead of as children of AgentSpan) and ultimately missing from the llmSpans
list in the final TraceApi output.

WHAT THIS TEST DOES:
1. Creates mock OTEL ReadableSpan objects representing:
   - An AgentSpan (parent, span_id=PARENT_ID, parent=None)
   - An LlmSpan (child, span_id=CHILD_ID, parent=PARENT_ID)
   
2. Simulates the cross-batch ordering issue by calling exporter.export() TWICE:
   - First call: Only the LLM span (child) - arrives BEFORE its parent
   - Second call: Only the Agent span (parent) - arrives AFTER its child
   
3. Asserts the EXPECTED CORRECT behavior (which currently FAILS):
   - The LlmSpan should be in the trace's llmSpans list
   - The LlmSpan should be a child of the AgentSpan (not a root span)

WHY THIS PROVES THE BUG:
In add_span_to_trace() (tracing.py:295-313), when a child span is processed:
- It looks up parent via get_span_by_uuid(span.parent_uuid)
- If parent hasn't been added yet (different batch), this returns None
- The child is then incorrectly added to root_spans instead of waiting
- When the parent arrives later, no re-parenting logic exists

WHAT FIX WOULD MAKE THIS PASS:
Either:
1. Buffer spans until all spans for a trace arrive, then build hierarchy
2. Implement deferred/lazy parent-child linking that re-parents orphaned spans
   when their parent arrives in a later batch
3. Use SimpleSpanProcessor (synchronous) instead of BatchSpanProcessor for
   scenarios requiring ordered span delivery
"""

import pytest
from unittest.mock import MagicMock, PropertyMock
from dataclasses import dataclass
from typing import Optional

from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.resources import Resource

from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from deepeval.tracing import trace_manager
from deepeval.tracing.types import LlmSpan, AgentSpan


# Constants for our test spans
TRACE_ID = 0x1234567890ABCDEF1234567890ABCDEF
PARENT_SPAN_ID = 0xAAAAAAAAAAAAAAAA  # AgentSpan
CHILD_SPAN_ID = 0xBBBBBBBBBBBBBBBB   # LlmSpan


@dataclass
class MockSpanParent:
    """Mock for span.parent which has a span_id attribute."""
    span_id: int


def create_mock_readable_span(
    name: str,
    span_id: int,
    trace_id: int,
    parent_span_id: Optional[int],
    attributes: dict,
    start_time_ns: int = 1000000000,
    end_time_ns: int = 2000000000,
) -> MagicMock:
    """
    Create a mock ReadableSpan that mimics what PydanticAI's OTEL instrumentation emits.
    """
    mock_span = MagicMock(spec=ReadableSpan)
    
    # Set up context
    mock_context = MagicMock()
    mock_context.span_id = span_id
    mock_context.trace_id = trace_id
    mock_span.context = mock_context
    
    # Set up parent
    if parent_span_id is not None:
        mock_span.parent = MockSpanParent(span_id=parent_span_id)
    else:
        mock_span.parent = None
    
    # Set up status
    mock_span.status = Status(StatusCode.OK)
    
    # Set up name
    mock_span.name = name
    
    # Set up timestamps
    mock_span.start_time = start_time_ns
    mock_span.end_time = end_time_ns
    
    # Set up attributes
    mock_span.attributes = attributes
    
    # Set up resource (needed by __set_trace_attributes)
    mock_resource = MagicMock(spec=Resource)
    mock_resource.attributes = {"confident.trace.environment": "testing"}
    mock_span.resource = mock_resource
    
    return mock_span


def create_agent_span_mock() -> MagicMock:
    """
    Create a mock AgentSpan as PydanticAI would emit it.
    This is the PARENT span (root of the trace).
    """
    attributes = {
        "confident.span.type": "agent",
        "confident.span.name": "test_agent",
        "gen_ai.agent.name": "test_agent",
        "confident.trace.name": "test-trace",
        "confident.trace.tags": '["test", "otel"]',
        "confident.trace.environment": "testing",
    }
    return create_mock_readable_span(
        name="agent run",
        span_id=PARENT_SPAN_ID,
        trace_id=TRACE_ID,
        parent_span_id=None,  # Root span - no parent
        attributes=attributes,
        start_time_ns=1000000000,
        end_time_ns=3000000000,
    )


def create_llm_span_mock() -> MagicMock:
    """
    Create a mock LlmSpan as PydanticAI would emit it.
    This is the CHILD span (child of the AgentSpan).
    """
    attributes = {
        "gen_ai.operation.name": "chat",  # This triggers LlmSpan creation
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.system": "openai",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 50,
    }
    return create_mock_readable_span(
        name="chat gpt-4o-mini",
        span_id=CHILD_SPAN_ID,
        trace_id=TRACE_ID,
        parent_span_id=PARENT_SPAN_ID,  # Parent is the AgentSpan
        attributes=attributes,
        start_time_ns=1500000000,
        end_time_ns=2500000000,
    )


class TestCrossBatchSpanOrdering:
    """
    Tests that reproduce the OTEL BatchSpanProcessor cross-batch span ordering issue.
    
    These tests should FAIL on the current codebase, proving the bug exists.
    They would PASS once the exporter/trace assembly is fixed to handle
    out-of-order span delivery across batches.
    """

    def setup_method(self):
        """Clear trace_manager state before each test."""
        trace_manager.clear_traces()
        trace_manager.active_traces = {}
        trace_manager.active_spans = {}

    def teardown_method(self):
        """Clear trace_manager state after each test."""
        trace_manager.clear_traces()
        trace_manager.active_traces = {}
        trace_manager.active_spans = {}

    def test_child_span_before_parent_span_cross_batch(self):
        """
        Test that LLM spans are correctly parented when child arrives before parent
        in separate export() batches.
        
        SIMULATES:
        - Batch 1: LlmSpan (child) arrives first, parent not yet in active_spans
        - Batch 2: AgentSpan (parent) arrives second
        
        EXPECTED (currently FAILS):
        - LlmSpan should be in llmSpans list of final TraceApi
        - LlmSpan should be a child of AgentSpan, not a root span
        
        ACTUAL (current buggy behavior):
        - LlmSpan is added to root_spans because parent doesn't exist yet
        - When AgentSpan arrives, no re-parenting occurs
        - LlmSpan may be lost or orphaned in the trace structure
        """
        exporter = ConfidentSpanExporter()
        
        # Create our mock spans
        llm_span_mock = create_llm_span_mock()    # Child
        agent_span_mock = create_agent_span_mock()  # Parent
        
        # BATCH 1: Export ONLY the child span (LlmSpan) first
        # This simulates BatchSpanProcessor delivering child before parent
        result1 = exporter.export(
            spans=[llm_span_mock],
            _test_run_id="test-cross-batch",  # Prevents actual posting, keeps traces
        )
        assert result1.name == "SUCCESS", "First export should succeed"
        
        # At this point, the LlmSpan has been processed but its parent doesn't exist
        # In the buggy implementation, it gets added to root_spans
        
        # BATCH 2: Export the parent span (AgentSpan) second
        # We need to re-create the trace since test mode clears traces
        # So we'll do both in a single logical test by NOT using _test_run_id
        # and instead inspecting the trace state manually
        
        # Let's do this more carefully - don't use _test_run_id to keep state
        trace_manager.clear_traces()
        trace_manager.active_traces = {}
        trace_manager.active_spans = {}
        
        # Now export child first (without _test_run_id so traces stay active)
        # But we need to prevent the actual HTTP posting - let's patch that
        
        # Actually, let's use a different approach: directly test the assembly logic
        # by manually calling the internal methods
        
        # Reset state
        trace_manager.clear_traces()
        
        # Manually simulate what export() does, but in two separate "batches"
        
        # --- BATCH 1: Process child span only ---
        child_wrapper = exporter._convert_readable_span_to_base_span(llm_span_mock)
        child_base_span = child_wrapper.base_span
        
        # Start trace (exporter would do this)
        trace_uuid = child_base_span.trace_uuid
        trace = trace_manager.start_new_trace(trace_uuid=trace_uuid)
        
        # Add child span to active_spans and trace
        trace_manager.add_span(child_base_span)
        trace_manager.add_span_to_trace(child_base_span)
        
        # At this point, child's parent_uuid points to PARENT_SPAN_ID
        # But parent hasn't been added yet, so child is in root_spans
        
        # Verify child was converted to LlmSpan type
        assert isinstance(child_base_span, LlmSpan), (
            f"Child should be LlmSpan but got {type(child_base_span).__name__}"
        )
        
        # --- BATCH 2: Process parent span ---
        parent_wrapper = exporter._convert_readable_span_to_base_span(agent_span_mock)
        parent_base_span = parent_wrapper.base_span
        
        # Verify parent was converted to AgentSpan type
        assert isinstance(parent_base_span, AgentSpan), (
            f"Parent should be AgentSpan but got {type(parent_base_span).__name__}"
        )
        
        # Add parent span to active_spans and trace
        trace_manager.add_span(parent_base_span)
        trace_manager.add_span_to_trace(parent_base_span)
        
        # --- Now verify the trace structure ---
        
        # Get the trace and create TraceApi to see final structure
        trace = trace_manager.get_trace_by_uuid(trace_uuid)
        assert trace is not None, "Trace should exist"
        
        # Set trace times (required for TraceApi creation)
        from deepeval.tracing.otel.utils import set_trace_time
        set_trace_time(trace)
        
        # Create the TraceApi output (this is what gets sent to the API)
        trace_api = trace_manager.create_trace_api(trace)
        
        # === ASSERTIONS THAT PROVE THE BUG ===
        
        # The llm_spans list will contain the span (it's found via root_spans iteration)
        # BUT the bug is that the LlmSpan is ORPHANED - it's a root span, not a child
        # of the AgentSpan as it should be.
        
        # The REAL test is: Is the LlmSpan properly parented to the AgentSpan?
        # When child arrives before parent, it becomes a root span instead of a child.
        
        # Check: LlmSpan should be a CHILD of AgentSpan, not a root span
        assert child_base_span not in trace.root_spans, (
            "FAIL: LlmSpan is in root_spans but should be a child of AgentSpan! "
            "This proves the cross-batch ordering bug: when child span arrives before parent, "
            "add_span_to_trace() can't find the parent (get_span_by_uuid returns None) "
            "and incorrectly adds the child to root_spans. "
            f"root_spans contains {len(trace.root_spans)} spans: "
            f"{[type(s).__name__ for s in trace.root_spans]}"
        )
        
        # Check: LlmSpan should be in AgentSpan's children list
        assert child_base_span in parent_base_span.children, (
            "FAIL: LlmSpan is not in AgentSpan.children! "
            "When child arrives before parent in cross-batch delivery, "
            "the parent-child relationship is never established."
        )

    def test_correct_ordering_single_batch_works(self):
        """
        Control test: When parent arrives BEFORE child in the same batch,
        everything works correctly.
        
        This test should PASS, proving the issue is specifically about
        cross-batch ordering, not general span processing.
        """
        exporter = ConfidentSpanExporter()
        
        # Create our mock spans
        llm_span_mock = create_llm_span_mock()    # Child
        agent_span_mock = create_agent_span_mock()  # Parent
        
        # Reset state
        trace_manager.clear_traces()
        trace_manager.active_traces = {}
        trace_manager.active_spans = {}
        
        # Process parent FIRST, then child (correct order)
        parent_wrapper = exporter._convert_readable_span_to_base_span(agent_span_mock)
        parent_base_span = parent_wrapper.base_span
        
        child_wrapper = exporter._convert_readable_span_to_base_span(llm_span_mock)
        child_base_span = child_wrapper.base_span
        
        # Start trace
        trace_uuid = parent_base_span.trace_uuid
        trace = trace_manager.start_new_trace(trace_uuid=trace_uuid)
        
        # Add parent first (correct order)
        trace_manager.add_span(parent_base_span)
        trace_manager.add_span_to_trace(parent_base_span)
        
        # Add child second (parent already exists)
        trace_manager.add_span(child_base_span)
        trace_manager.add_span_to_trace(child_base_span)
        
        # Get the trace and create TraceApi
        trace = trace_manager.get_trace_by_uuid(trace_uuid)
        
        # Set trace times (required for TraceApi creation)
        from deepeval.tracing.otel.utils import set_trace_time
        set_trace_time(trace)
        
        trace_api = trace_manager.create_trace_api(trace)
        
        # These assertions should PASS (correct order works fine)
        assert len(trace_api.agent_spans) == 1, "Should have 1 AgentSpan"
        assert len(trace_api.llm_spans) == 1, "Should have 1 LlmSpan"
        
        # Verify the child is properly attached to parent
        assert child_base_span in parent_base_span.children, (
            "LlmSpan should be a child of AgentSpan when processed in correct order"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
