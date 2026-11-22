"""Regression tests for deepeval #1410: the non-exact tool-correctness reason
reports 'missing tools [...]' with an empty list when argument-level scores
differ but every expected tool was in fact called."""
import pytest

from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from deepeval.test_case import ToolCall, ToolCallParams


def _metric(called, expected):
    """Build the metric without wiring a judge model."""
    m = ToolCorrectnessMetric.__new__(ToolCorrectnessMetric)
    m.tools_called = called
    m.expected_tools = expected
    m.evaluation_params = [ToolCallParams.INPUT_PARAMETERS]
    m.should_exact_match = False
    m.should_consider_ordering = False
    return m


def test_parameter_mismatch_does_not_claim_missing_tools():
    """Every expected tool was called; only the arguments differ."""
    m = _metric(
        [ToolCall(name="WebSearch", input_parameters={"query": "running shoes"})],
        [ToolCall(name="WebSearch", input_parameters={"query": "trail running shoes"})],
    )
    assert m._calculate_non_exact_match_score() < 1
    reason = m._generate_reason()
    assert "missing tools" not in reason
    assert "argument" in reason.lower()
    assert "Incomplete tool usage" in reason


def test_genuinely_missing_tool_still_reported():
    """Guard the opposite direction: a real absence must still be reported."""
    m = _metric(
        [ToolCall(name="WebSearch", input_parameters={})],
        [ToolCall(name="WebSearch", input_parameters={}),
         ToolCall(name="DatabaseQuery", input_parameters={})],
    )
    reason = m._generate_reason()
    assert "missing tools" in reason
    assert "DatabaseQuery" in reason
