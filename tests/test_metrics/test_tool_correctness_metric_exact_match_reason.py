"""Deterministic regression tests for deepeval#3245.

ToolCorrectnessMetric reported phantom tool-type mismatches in exact-match
mode: _calculate_exact_match_score() compares corresponding positions, but
_get_type_mismatches() compared every same-named pair across positions. These
tests exercise the deterministic helpers directly (no LLM, no API key).
"""

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCall, ToolCallType


def build_metric(expected, called, exact_match=True):
    metric = ToolCorrectnessMetric.__new__(ToolCorrectnessMetric)
    metric.should_exact_match = exact_match
    metric.should_consider_ordering = False
    metric.evaluation_params = []
    metric.expected_tools = expected
    metric.tools_called = called
    return metric


def call(name, type_):
    return ToolCall(name=name, type=type_)


class TestExactMatchTypeMismatchDiagnostics:
    def test_identical_mixed_type_sequence_has_no_phantom_mismatches(self):
        # Issue #3245's exact repro: same tool name with different types at
        # corresponding positions is an exact match with no diagnostics.
        expected = [
            call("search", ToolCallType.FUNCTION),
            call("search", ToolCallType.MCP),
        ]
        metric = build_metric(expected, [c.model_copy() for c in expected])

        assert metric._calculate_exact_match_score() == 1.0
        reason = metric._generate_reason()
        assert reason.startswith("Exact match")
        assert "Tool type mismatches" not in reason

    def test_genuine_type_mismatch_still_reported_at_position(self):
        expected = [call("search", ToolCallType.FUNCTION)]
        called = [call("search", ToolCallType.MCP)]
        metric = build_metric(expected, called)

        assert metric._calculate_exact_match_score() == 0.0
        reason = metric._generate_reason()
        assert reason.startswith("Not an exact match")
        assert "Tool type mismatches" in reason
        assert "search (expected FUNCTION, called MCP)" in reason

    def test_swapped_types_reported_per_position(self):
        # Both positions genuinely mismatch; both must be reported, once each.
        expected = [
            call("search", ToolCallType.FUNCTION),
            call("search", ToolCallType.MCP),
        ]
        called = [
            call("search", ToolCallType.MCP),
            call("search", ToolCallType.FUNCTION),
        ]
        metric = build_metric(expected, called)

        assert metric._calculate_exact_match_score() == 0.0
        reason = metric._generate_reason()
        assert "Tool type mismatches" in reason
        assert "search (expected FUNCTION, called MCP)" in reason
        assert "search (expected MCP, called FUNCTION)" in reason

    def test_missing_call_has_no_phantom_mismatches(self):
        expected = [
            call("search", ToolCallType.FUNCTION),
            call("search", ToolCallType.MCP),
        ]
        called = [call("search", ToolCallType.FUNCTION)]
        metric = build_metric(expected, called)

        assert metric._calculate_exact_match_score() == 0.0
        reason = metric._generate_reason()
        assert reason.startswith("Not an exact match")
        assert "Tool type mismatches" not in reason

    def test_name_mismatch_is_not_a_type_mismatch(self):
        expected = [call("search", ToolCallType.FUNCTION)]
        called = [call("lookup", ToolCallType.MCP)]
        metric = build_metric(expected, called)

        assert metric._calculate_exact_match_score() == 0.0
        assert "Tool type mismatches" not in metric._generate_reason()


class TestNonExactMatchBehaviorUnchanged:
    def test_any_pair_mismatches_preserved_outside_exact_match(self):
        # Non-exact modes keep the original any-pair diagnostic behavior.
        expected = [
            call("search", ToolCallType.FUNCTION),
            call("search", ToolCallType.MCP),
        ]
        called = [
            call("search", ToolCallType.FUNCTION),
            call("search", ToolCallType.MCP),
        ]
        metric = build_metric(expected, called, exact_match=False)

        mismatches = metric._get_type_mismatches()
        assert "search (expected FUNCTION, called MCP)" in mismatches
        assert "search (expected MCP, called FUNCTION)" in mismatches
