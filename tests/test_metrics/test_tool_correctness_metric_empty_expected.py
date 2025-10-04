import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall


def test_tool_correctness_empty_expected_and_called():
    metric = ToolCorrectnessMetric()
    test_case = LLMTestCase(
        input="What is an elephant?",
        actual_output="...",
        tools_called=[],
        expected_tools=[],
    )
    metric.measure(test_case)
    assert metric.score == 1.0, f"Expected score 1.0, got {metric.score}"


def test_tool_correctness_empty_expected_nonempty_called():
    metric = ToolCorrectnessMetric()
    tool_call = ToolCall(name="search", input_parameters={}, output=None)
    test_case = LLMTestCase(
        input="What is an elephant?",
        actual_output="...",
        tools_called=[tool_call],
        expected_tools=[],
    )
    metric.measure(test_case)
    assert metric.score == 0.0, f"Expected score 0.0, got {metric.score}"
