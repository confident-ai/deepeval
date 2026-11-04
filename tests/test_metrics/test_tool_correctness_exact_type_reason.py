import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCall


@pytest.mark.parametrize(
    "called_types, expected_score, expected_mismatches",
    [
        (["FUNCTION", "MCP"], 1.0, []),
        (
            ["MCP", "FUNCTION"],
            0.0,
            [
                "search (expected FUNCTION, called MCP)",
                "search (expected MCP, called FUNCTION)",
            ],
        ),
        (
            ["FUNCTION", "FUNCTION"],
            0.0,
            ["search (expected MCP, called FUNCTION)"],
        ),
        (["FUNCTION"], 0.0, []),
        (["FUNCTION", "MCP", "FUNCTION"], 0.0, []),
    ],
)
def test_exact_match_type_reason_uses_corresponding_calls(
    called_types, expected_score, expected_mismatches
):
    # Exercise deterministic helpers without initializing an LLM provider.
    metric = ToolCorrectnessMetric.__new__(ToolCorrectnessMetric)
    metric.should_exact_match = True
    metric.evaluation_params = []
    metric.expected_tools = [
        ToolCall(name="search", type=tool_type)
        for tool_type in ["FUNCTION", "MCP"]
    ]
    metric.tools_called = [
        ToolCall(name="search", type=tool_type) for tool_type in called_types
    ]

    assert metric._calculate_exact_match_score() == expected_score
    assert metric._get_type_mismatches() == expected_mismatches
    reason = metric._generate_reason()
    assert ("Tool type mismatches" in reason) == bool(expected_mismatches)
