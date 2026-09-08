"""Tests for CostEfficiencyMetric.

All tests are fully deterministic — no API key, no network, no LLM required.
"""

import pytest

from deepeval.test_case import LLMTestCase
from deepeval.metrics.cost_efficiency import CostEfficiencyMetric


# ---------------------------------------------------------------------------
# Trace / test-case builder helpers
# ---------------------------------------------------------------------------


def _make_llm_span(input_tokens=100, output_tokens=50, children=None) -> dict:
    """Build a minimal LLM span dict with token counts."""
    return {
        "type": "llm",
        "name": "llm_call",
        "input": "some prompt",
        "output": "some response",
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "children": children or [],
    }


def _make_agent_span(children: list) -> dict:
    """Build a root agent span wrapping child spans."""
    return {
        "type": "agent",
        "name": "agent",
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
# Test 1: within budget → score 1.0
# ---------------------------------------------------------------------------


def test_within_budget_scores_one():
    trace = _make_agent_span(
        [_make_llm_span(input_tokens=100, output_tokens=50)]
    )
    metric = CostEfficiencyMetric(token_budget=1000)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 1.0
    assert metric.success is True
    assert metric.score_breakdown["total_tokens"] == 150


# ---------------------------------------------------------------------------
# Test 2: over budget → proportional score
# ---------------------------------------------------------------------------


def test_over_budget_scores_proportionally():
    # 3000 total tokens against a 1000 budget → 1/3 ≈ 0.333 (< threshold 0.5)
    trace = _make_agent_span(
        [_make_llm_span(input_tokens=2500, output_tokens=500)]
    )
    metric = CostEfficiencyMetric(token_budget=1000)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == pytest.approx(1 / 3)
    assert metric.success is False
    assert "exceeded" in metric.reason


# ---------------------------------------------------------------------------
# Test 3: exactly at budget → score 1.0
# ---------------------------------------------------------------------------


def test_exact_budget_scores_one():
    trace = _make_agent_span(
        [_make_llm_span(input_tokens=200, output_tokens=100)]
    )
    metric = CostEfficiencyMetric(token_budget=300)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 1.0


# ---------------------------------------------------------------------------
# Test 4: nested trace sums every LLM span
# ---------------------------------------------------------------------------


def test_nested_trace_sums_all_llm_spans():
    # Two LLM spans nested at different depths → 100+50 + 200+100 = 450
    trace = _make_agent_span(
        [
            _make_llm_span(input_tokens=100, output_tokens=50),
            _make_llm_span(
                input_tokens=200,
                output_tokens=100,
                children=[_make_llm_span(input_tokens=0, output_tokens=0)],
            ),
        ]
    )
    metric = CostEfficiencyMetric(token_budget=1000)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score_breakdown["total_tokens"] == 450
    assert metric.score_breakdown["llm_span_count"] == 3


# ---------------------------------------------------------------------------
# Test 5: no LLM spans → 0 tokens → score 1.0
# ---------------------------------------------------------------------------


def test_no_llm_spans_scores_one():
    trace = _make_agent_span([])
    metric = CostEfficiencyMetric(token_budget=1000)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 1.0
    assert metric.score_breakdown["total_tokens"] == 0
    assert "No LLM token usage" in metric.reason


# ---------------------------------------------------------------------------
# Test 6: missing trace → score 0.0 with descriptive reason
# ---------------------------------------------------------------------------


def test_no_trace_returns_zero():
    metric = CostEfficiencyMetric(token_budget=1000)
    tc = LLMTestCase(input="x", actual_output="y")
    tc._trace_dict = None
    metric._calculate_metric(tc)

    assert metric.score == 0.0
    assert "No trace data" in metric.reason


# ---------------------------------------------------------------------------
# Test 7: strict mode zeroes any score below threshold
# ---------------------------------------------------------------------------


def test_strict_mode_zeroes_below_threshold():
    trace = _make_agent_span(
        [_make_llm_span(input_tokens=1500, output_tokens=500)]
    )
    metric = CostEfficiencyMetric(token_budget=1000, strict_mode=True)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 0.0


# ---------------------------------------------------------------------------
# Test 8: invalid budget is rejected at construction
# ---------------------------------------------------------------------------


def test_non_positive_budget_raises():
    with pytest.raises(ValueError):
        CostEfficiencyMetric(token_budget=0)
    with pytest.raises(ValueError):
        CostEfficiencyMetric(token_budget=-5)
