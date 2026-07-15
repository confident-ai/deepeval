"""Tests for TokenBudgetMetric.

All tests are fully deterministic — no API key, no network, no LLM required.
"""

import pytest

from deepeval.test_case import LLMTestCase
from deepeval.metrics.token_budget import TokenBudgetMetric

# ---------------------------------------------------------------------------
# Trace / test-case builder helpers
# ---------------------------------------------------------------------------


def _make_llm_span(
    name: str = "llm_call",
    input_tokens=None,
    output_tokens=None,
    cost_per_input=None,
    cost_per_output=None,
    children=None,
) -> dict:
    """Build an LLM span dict matching DeepEval's _trace_dict format.

    Token/cost fields are only included when non-None, mirroring
    ``create_nested_spans_dict`` which drops all None-valued keys.
    """
    span = {
        "type": "llm",
        "name": name,
        "input": "some prompt",
        "output": "some completion",
        "children": children or [],
    }
    if input_tokens is not None:
        span["input_token_count"] = input_tokens
    if output_tokens is not None:
        span["output_token_count"] = output_tokens
    if cost_per_input is not None:
        span["cost_per_input_token"] = cost_per_input
    if cost_per_output is not None:
        span["cost_per_output_token"] = cost_per_output
    return span


def _make_tool_span(name: str, input_data: dict, children=None) -> dict:
    return {
        "type": "tool",
        "name": name,
        "input": input_data,
        "output": f"result of {name}",
        "children": children or [],
    }


def _make_agent_span(name: str, children: list) -> dict:
    return {
        "type": "agent",
        "name": name,
        "input": "user query",
        "output": "agent answer",
        "children": children,
    }


def _make_test_case(trace_dict: dict) -> LLMTestCase:
    tc = LLMTestCase(input="test input", actual_output="test output")
    tc._trace_dict = trace_dict
    return tc


# ---------------------------------------------------------------------------
# Test 1: Under budget → score 1.0
# ---------------------------------------------------------------------------


def test_under_budget_passes():
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(input_tokens=100, output_tokens=50),
            _make_llm_span(input_tokens=200, output_tokens=80),
        ],
    )
    metric = TokenBudgetMetric(max_total_tokens=1000)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 1.0
    assert metric.success is True
    assert metric.score_breakdown["total_tokens"]["spent"] == 430


# ---------------------------------------------------------------------------
# Test 2: Over total-token budget → graded sub-score < 1.0
# ---------------------------------------------------------------------------


def test_over_total_token_budget_fails():
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(input_tokens=600, output_tokens=600),
        ],
    )
    # 1200 spent vs 600 budget → exactly 2x over → sub-score 0.5
    metric = TokenBudgetMetric(max_total_tokens=600)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == pytest.approx(0.5)
    assert metric.success is False
    assert "total tokens" in metric.reason


# ---------------------------------------------------------------------------
# Test 3: Input-token budget only
# ---------------------------------------------------------------------------


def test_input_token_budget_only():
    trace = _make_agent_span(
        "agent",
        [_make_llm_span(input_tokens=500, output_tokens=10000)],
    )
    # Only input tokens are gated; huge output must not affect the score.
    metric = TokenBudgetMetric(max_input_tokens=1000)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 1.0
    assert "output_tokens" not in metric.score_breakdown


# ---------------------------------------------------------------------------
# Test 4: Output-token budget only
# ---------------------------------------------------------------------------


def test_output_token_budget_only():
    trace = _make_agent_span(
        "agent",
        [_make_llm_span(input_tokens=10000, output_tokens=2000)],
    )
    # 2000 output vs 1000 budget → 2x over → 0.5
    metric = TokenBudgetMetric(max_output_tokens=1000)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == pytest.approx(0.5)
    assert "input_tokens" not in metric.score_breakdown


# ---------------------------------------------------------------------------
# Test 5: Cost budget with per-token rates
# ---------------------------------------------------------------------------


def test_cost_budget():
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(
                input_tokens=1000,
                output_tokens=500,
                cost_per_input=0.00001,
                cost_per_output=0.00003,
            ),
        ],
    )
    # cost = 1000*1e-5 + 500*3e-5 = 0.01 + 0.015 = 0.025
    metric = TokenBudgetMetric(max_cost=0.05)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 1.0
    assert metric.score_breakdown["cost"]["spent"] == pytest.approx(0.025)


def test_cost_budget_exceeded():
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(
                input_tokens=1000,
                output_tokens=1000,
                cost_per_input=0.00002,
                cost_per_output=0.00002,
            ),
        ],
    )
    # cost = 0.04; budget 0.02 → 2x over → 0.5
    metric = TokenBudgetMetric(max_cost=0.02)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == pytest.approx(0.5)
    assert metric.success is False
    assert "cost" in metric.reason


# ---------------------------------------------------------------------------
# Test 6: Most-expensive-span flag
# ---------------------------------------------------------------------------


def test_most_expensive_span_flagged():
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(
                name="cheap",
                input_tokens=10,
                output_tokens=10,
                cost_per_input=0.000001,
                cost_per_output=0.000001,
            ),
            _make_llm_span(
                name="expensive",
                input_tokens=1000,
                output_tokens=1000,
                cost_per_input=0.00001,
                cost_per_output=0.00001,
            ),
        ],
    )
    metric = TokenBudgetMetric(max_cost=1.0)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score_breakdown["most_expensive_span"]["name"] == "expensive"


# ---------------------------------------------------------------------------
# Test 7: Null token counts → counted, not crashed
# ---------------------------------------------------------------------------


def test_null_token_counts_reported():
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(input_tokens=100, output_tokens=50),
            _make_llm_span(),  # no token data at all
        ],
    )
    metric = TokenBudgetMetric(max_total_tokens=1000)
    metric._calculate_metric(_make_test_case(trace))

    # Only the tokened span contributes; the null one is counted, not summed.
    assert metric.score_breakdown["total_tokens"]["spent"] == 150
    assert metric.score_breakdown["untokened_spans"] == 1
    assert "no token data" in metric.reason


# ---------------------------------------------------------------------------
# Test 8: Tokens present but no pricing → unpriced, excluded from cost
# ---------------------------------------------------------------------------


def test_unpriced_spans_excluded_from_cost():
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(input_tokens=1000, output_tokens=1000),  # no rates
        ],
    )
    metric = TokenBudgetMetric(max_cost=0.05)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score_breakdown["cost"]["spent"] == 0.0
    assert metric.score_breakdown["unpriced_spans"] == 1
    assert "no per-token pricing" in metric.reason


# ---------------------------------------------------------------------------
# Test 9: strict_mode → binary score
# ---------------------------------------------------------------------------


def test_strict_mode_binary():
    trace = _make_agent_span(
        "agent",
        [_make_llm_span(input_tokens=600, output_tokens=200)],
    )
    # 800 spent vs 1000 budget → graded would be 1.0 anyway; make it breach:
    metric = TokenBudgetMetric(max_total_tokens=500, strict_mode=True)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 0.0
    assert metric.threshold == 1.0

    # And a passing run under strict_mode is exactly 1.0
    metric_ok = TokenBudgetMetric(max_total_tokens=2000, strict_mode=True)
    metric_ok._calculate_metric(_make_test_case(trace))
    assert metric_ok.score == 1.0


# ---------------------------------------------------------------------------
# Test 10: Minimum across multiple budgets is the gate
# ---------------------------------------------------------------------------


def test_min_across_budgets():
    trace = _make_agent_span(
        "agent",
        [_make_llm_span(input_tokens=100, output_tokens=400)],
    )
    # total=500 (under 1000 → 1.0); output=400 vs 200 → 2x over → 0.5.
    # score must be the min = 0.5.
    metric = TokenBudgetMetric(max_total_tokens=1000, max_output_tokens=200)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == pytest.approx(0.5)
    assert metric.score_breakdown["total_tokens"]["sub_score"] == 1.0
    assert metric.score_breakdown["output_tokens"][
        "sub_score"
    ] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 11: All-None budgets → ValueError
# ---------------------------------------------------------------------------


def test_no_budget_raises():
    with pytest.raises(ValueError):
        TokenBudgetMetric()


# ---------------------------------------------------------------------------
# Test 12: No trace data → score 0.0 with descriptive reason
# ---------------------------------------------------------------------------


def test_no_trace_returns_zero():
    metric = TokenBudgetMetric(max_total_tokens=1000)
    tc = LLMTestCase(input="x", actual_output="y")
    tc._trace_dict = None
    metric._calculate_metric(tc)

    assert metric.score == 0.0
    assert "No trace data" in metric.reason


# ---------------------------------------------------------------------------
# Test 13: Nested spans are walked recursively
# ---------------------------------------------------------------------------


def test_nested_spans_walked():
    deep_llm = _make_llm_span(input_tokens=300, output_tokens=100)
    tool = _make_tool_span("search", {"q": "x"}, children=[deep_llm])
    trace = _make_agent_span(
        "agent",
        [
            _make_llm_span(input_tokens=100, output_tokens=50),
            tool,  # LLM span buried under a tool span
        ],
    )
    metric = TokenBudgetMetric(max_total_tokens=10000)
    metric._calculate_metric(_make_test_case(trace))

    # Both LLM spans counted: 150 + 400 = 550
    assert metric.score_breakdown["total_tokens"]["spent"] == 550
    assert metric.score_breakdown["llm_span_count"] == 2


# ---------------------------------------------------------------------------
# Test 14: No LLM spans → 1.0 but reason must NOT claim the budget was respected
# ---------------------------------------------------------------------------


def test_no_llm_spans_reports_unmeasurable():
    """A trace with tool/agent spans but zero LLM spans cannot have its token
    budget measured. The score stays 1.0 (an agent may legitimately make no
    LLM calls), but the reason must say the budget was unmeasurable rather
    than falsely claiming it was respected."""
    trace = _make_agent_span(
        "agent",
        [_make_tool_span("search", {"q": "x"})],  # no LLM span anywhere
    )
    metric = TokenBudgetMetric(max_total_tokens=1000, max_cost=0.05)
    metric._calculate_metric(_make_test_case(trace))

    assert metric.score == 1.0
    assert metric.score_breakdown["llm_span_count"] == 0
    assert "could not be measured" in metric.reason
    assert "respected" not in metric.reason
