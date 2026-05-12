"""Tests for AgentLoopDetectionMetric.

All tests are fully deterministic — no API key, no network, no LLM required.
"""

import pytest

from deepeval.test_case import LLMTestCase
from deepeval.metrics.agent_loop_detection import AgentLoopDetectionMetric


# ---------------------------------------------------------------------------
# Trace / test-case builder helpers
# ---------------------------------------------------------------------------


def _make_tool_span(name: str, input_data: dict, children=None) -> dict:
    """Build a minimal tool span dict matching DeepEval's _trace_dict format."""
    span = {
        "type": "tool",
        "name": name,
        "input": input_data,
        "output": f"result of {name}",
        "children": children or [],
    }
    return span


def _make_llm_span(output: str, children=None) -> dict:
    """Build a minimal LLM span dict."""
    return {
        "type": "llm",
        "name": "llm_call",
        "input": "some prompt",
        "output": output,
        "children": children or [],
    }


def _make_agent_span(name: str, children: list) -> dict:
    """Build a root agent span wrapping child spans."""
    return {
        "type": "agent",
        "name": name,
        "input": "user query",
        "output": "agent answer",
        "children": children,
    }


def _make_test_case(trace_dict: dict) -> LLMTestCase:
    """Create an LLMTestCase with a pre-built _trace_dict (no API needed)."""
    tc = LLMTestCase(input="test input", actual_output="test output")
    tc._trace_dict = trace_dict
    return tc


# ---------------------------------------------------------------------------
# Test 1: Clean trace → score 1.0
# ---------------------------------------------------------------------------


def test_clean_trace_passes():
    """A well-behaved agent with distinct tool calls should score 1.0."""
    trace = _make_agent_span(
        "planner",
        [
            _make_tool_span("search_web", {"query": "Paris weather"}),
            _make_tool_span("get_forecast", {"city": "Paris"}),
            _make_tool_span("summarize", {"text": "It is sunny"}),
        ],
    )
    metric = AgentLoopDetectionMetric(threshold=0.5)
    tc = _make_test_case(trace)
    metric._calculate_metric(tc)

    assert metric.score == 1.0
    assert metric.success is True


# ---------------------------------------------------------------------------
# Test 2: Repeated tool calls → score ≤ 0.5
# ---------------------------------------------------------------------------


def test_repeated_tool_calls_detected():
    """Four identical tool calls with repetition_threshold=3 should fail.

    The metric scores 0.5 for moderate repetition (count >= threshold but
    < 2x threshold).  Using threshold=0.7 ensures 0.5 < threshold → failure.
    """
    trace = _make_agent_span(
        "looping_agent",
        [
            _make_tool_span("search_web", {"query": "Paris weather"}),
            _make_tool_span("search_web", {"query": "Paris weather"}),
            _make_tool_span("search_web", {"query": "Paris weather"}),
            _make_tool_span("search_web", {"query": "Paris weather"}),
        ],
    )
    metric = AgentLoopDetectionMetric(
        threshold=0.7,  # > 0.5 so that a moderate repetition score fails
        repetition_threshold=3,
        check_reasoning_stagnation=False,
        check_call_graph_cycles=False,
    )
    tc = _make_test_case(trace)
    metric._calculate_metric(tc)

    assert metric.score_breakdown["tool_repetition"] <= 0.5
    assert metric.success is False


# ---------------------------------------------------------------------------
# Test 3: No trace data → score 0.0 with "No trace data" in reason
# ---------------------------------------------------------------------------


def test_no_trace_returns_zero():
    """Missing trace should immediately return 0.0 with a descriptive reason."""
    metric = AgentLoopDetectionMetric()
    tc = LLMTestCase(input="x", actual_output="y")
    tc._trace_dict = None
    metric._calculate_metric(tc)

    assert metric.score == 0.0
    assert "No trace data" in metric.reason


# ---------------------------------------------------------------------------
# Test 4: Stagnating LLM outputs → stagnation score < 1.0
# ---------------------------------------------------------------------------


def test_reasoning_stagnation_detected():
    """Identical long LLM outputs should be flagged as stagnating."""
    repeated_output = (
        "I need to search for Paris weather using the search tool. "
        "The user wants current weather conditions in Paris France. "
        "I will call the search tool with query Paris weather forecast today. "
        "After retrieving results I will summarize them clearly for the user. "
        "Paris weather search results will help answer the question completely."
    )
    # Build a trace with 3 identical LLM outputs (all > 20 meaningful words)
    trace = _make_agent_span(
        "stagnating_agent",
        [
            _make_llm_span(repeated_output),
            _make_llm_span(repeated_output),
            _make_llm_span(repeated_output),
        ],
    )
    metric = AgentLoopDetectionMetric(
        similarity_threshold=0.75,
        check_tool_repetition=False,
        check_call_graph_cycles=False,
    )
    tc = _make_test_case(trace)
    metric._calculate_metric(tc)

    assert metric.score_breakdown["reasoning_stagnation"] < 1.0


# ---------------------------------------------------------------------------
# Test 5: Selective sub-signal disabling
# ---------------------------------------------------------------------------


def test_disable_tool_repetition_check():
    """When check_tool_repetition=False, repeated tools must not penalise score."""
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span("search_web", {"query": "q"}),
            _make_tool_span("search_web", {"query": "q"}),
            _make_tool_span("search_web", {"query": "q"}),
            _make_tool_span("search_web", {"query": "q"}),
        ],
    )
    metric = AgentLoopDetectionMetric(
        check_tool_repetition=False,
        check_reasoning_stagnation=False,
        check_call_graph_cycles=False,
    )
    tc = _make_test_case(trace)
    metric._calculate_metric(tc)

    # All checks disabled → should default to 1.0
    assert metric.score_breakdown["tool_repetition"] == 1.0
    assert metric.score == 1.0


# ---------------------------------------------------------------------------
# Test 6: Score combining weights normalise correctly
# ---------------------------------------------------------------------------


def test_score_combines_with_correct_weights():
    """When cycles are disabled, rep+stag weights must sum to 1.0 after norm."""
    metric = AgentLoopDetectionMetric(
        check_tool_repetition=True,
        check_reasoning_stagnation=True,
        check_call_graph_cycles=False,
    )
    # rep=1.0 * 0.40, stag=0.0 * 0.35 → total=0.40 / 0.75
    combined = metric._combine_scores(1.0, 0.0, 1.0)
    expected = 0.40 / 0.75
    assert abs(combined - expected) < 0.001


# ---------------------------------------------------------------------------
# Test 7: Call graph cycle (true recursive span) → cycle score 0.0
# ---------------------------------------------------------------------------


def test_call_graph_cycle_detected():
    """A span whose name+type appears on its own ancestry path is a cycle."""
    # agent → tool_A → agent  (agent appears as both ancestor and descendant)
    inner_agent = {
        "type": "agent",
        "name": "planner",
        "input": "recurse",
        "output": "stuck",
        "children": [],
    }
    outer_agent = {
        "type": "agent",
        "name": "planner",  # same label → cycle
        "input": "user query",
        "output": "answer",
        "children": [
            _make_tool_span("tool_a", {"x": "1"}, children=[inner_agent]),
        ],
    }
    metric = AgentLoopDetectionMetric(
        check_tool_repetition=False,
        check_reasoning_stagnation=False,
        check_call_graph_cycles=True,
    )
    tc = _make_test_case(outer_agent)
    metric._calculate_metric(tc)

    assert metric.score_breakdown["call_graph_cycles"] == 0.0
    assert metric.success is False
    assert "Cycle" in metric.reason


# ---------------------------------------------------------------------------
# Test 8: Sequential same-name calls are NOT flagged as call graph cycles
# ---------------------------------------------------------------------------


def test_sequential_same_name_not_a_cycle():
    """
    Calling tool A then tool B then tool A again is NOT a cycle — tool A appears
    twice at sibling positions, not on the same ancestry path.
    """
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span("search_web", {"q": "foo"}),
            _make_tool_span("other_tool", {"x": "bar"}),
            _make_tool_span(
                "search_web", {"q": "baz"}
            ),  # same name, different args
        ],
    )
    metric = AgentLoopDetectionMetric(
        check_tool_repetition=False,
        check_reasoning_stagnation=False,
        check_call_graph_cycles=True,
    )
    tc = _make_test_case(trace)
    metric._calculate_metric(tc)

    # No cycle — these are siblings, not ancestors of each other
    assert metric.score_breakdown["call_graph_cycles"] == 1.0
