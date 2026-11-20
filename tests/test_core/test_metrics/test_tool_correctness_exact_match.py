from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCall, ToolCallType


def _create_metric(
    expected_tools,
    tools_called,
    should_exact_match=True,
    should_consider_ordering=False,
    evaluation_params=None,
):
    metric = ToolCorrectnessMetric.__new__(ToolCorrectnessMetric)
    metric.should_exact_match = should_exact_match
    metric.should_consider_ordering = should_consider_ordering
    metric.strict_mode = False
    metric.threshold = 0.5
    metric.evaluation_params = evaluation_params or []
    metric.expected_tools = expected_tools
    metric.tools_called = tools_called
    return metric


def test_exact_match_identical_sequence_with_multiple_types_for_same_tool_name():
    expected = [
        ToolCall(name="search", type=ToolCallType.FUNCTION),
        ToolCall(name="search", type=ToolCallType.MCP),
    ]
    called = [
        ToolCall(name="search", type=ToolCallType.FUNCTION),
        ToolCall(name="search", type=ToolCallType.MCP),
    ]
    metric = _create_metric(expected, called, should_exact_match=True)

    score = metric._calculate_exact_match_score()
    mismatches = metric._get_type_mismatches()
    reason = metric._generate_reason()

    assert score == 1.0
    assert mismatches == []
    assert "Tool type mismatches" not in reason
    assert (
        reason
        == "Exact match: expected ['search', 'search'], called ['search', 'search']. See details above."
    )


def test_exact_match_genuine_type_mismatch():
    expected = [
        ToolCall(name="search", type=ToolCallType.FUNCTION),
        ToolCall(name="calc", type=ToolCallType.MCP),
    ]
    called = [
        ToolCall(name="search", type=ToolCallType.MCP),
        ToolCall(name="calc", type=ToolCallType.MCP),
    ]
    metric = _create_metric(expected, called, should_exact_match=True)

    score = metric._calculate_exact_match_score()
    mismatches = metric._get_type_mismatches()
    reason = metric._generate_reason()

    assert score == 0.0
    assert mismatches == ["search (expected FUNCTION, called MCP)"]
    assert (
        "Tool type mismatches: ['search (expected FUNCTION, called MCP)']"
        in reason
    )
    assert "Not an exact match" in reason


def test_exact_match_swapped_types():
    expected = [
        ToolCall(name="search", type=ToolCallType.FUNCTION),
        ToolCall(name="search", type=ToolCallType.MCP),
    ]
    called = [
        ToolCall(name="search", type=ToolCallType.MCP),
        ToolCall(name="search", type=ToolCallType.FUNCTION),
    ]
    metric = _create_metric(expected, called, should_exact_match=True)

    score = metric._calculate_exact_match_score()
    mismatches = metric._get_type_mismatches()
    reason = metric._generate_reason()

    assert score == 0.0
    assert mismatches == [
        "search (expected FUNCTION, called MCP)",
        "search (expected MCP, called FUNCTION)",
    ]
    assert "search (expected FUNCTION, called MCP)" in reason
    assert "search (expected MCP, called FUNCTION)" in reason


def test_exact_match_differing_sequence_lengths():
    expected = [
        ToolCall(name="search", type=ToolCallType.FUNCTION),
        ToolCall(name="calc", type=ToolCallType.FUNCTION),
    ]
    called = [
        ToolCall(name="search", type=ToolCallType.MCP),
    ]
    metric = _create_metric(expected, called, should_exact_match=True)

    score = metric._calculate_exact_match_score()
    mismatches = metric._get_type_mismatches()
    reason = metric._generate_reason()

    assert score == 0.0
    assert mismatches == ["search (expected FUNCTION, called MCP)"]
    assert (
        "Tool type mismatches: ['search (expected FUNCTION, called MCP)']"
        in reason
    )


def test_non_exact_match_mode_retains_behavior():
    expected = [
        ToolCall(name="search", type=ToolCallType.FUNCTION),
    ]
    called = [
        ToolCall(name="search", type=ToolCallType.MCP),
    ]
    metric = _create_metric(expected, called, should_exact_match=False)

    mismatches = metric._get_type_mismatches()
    assert mismatches == ["search (expected FUNCTION, called MCP)"]
