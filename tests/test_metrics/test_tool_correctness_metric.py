import os
import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallType

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)


def build_test_case(called_type: ToolCallType, expected_type: ToolCallType):
    return LLMTestCase(
        input="What does the confident-ai/deepeval repo use for metrics?",
        actual_output="ask_question({})",
        tools_called=[
            ToolCall(
                name="ask_question",
                type=called_type,
                input_parameters={"repoName": "confident-ai/deepeval"},
            )
        ],
        expected_tools=[
            ToolCall(
                name="ask_question",
                type=expected_type,
                input_parameters={"repoName": "confident-ai/deepeval"},
            )
        ],
    )


class TestToolCorrectnessMetricType:
    """Tests for tool call type matching in tool correctness metric"""

    def test_type_mismatch_fails(self):
        metric = ToolCorrectnessMetric(async_mode=False)
        metric.measure(
            build_test_case(ToolCallType.MCP, ToolCallType.FUNCTION),
            _show_indicator=False,
        )

        assert metric.score == 0.0
        assert metric.success is False
        assert "tool type mismatches" in metric.reason
        assert "expected FUNCTION, called MCP" in metric.reason

    def test_matching_type_passes(self):
        metric = ToolCorrectnessMetric(async_mode=False)
        metric.measure(
            build_test_case(ToolCallType.MCP, ToolCallType.MCP),
            _show_indicator=False,
        )

        assert metric.score == 1.0
        assert metric.success is True

    def test_type_mismatch_fails_exact_match(self):
        metric = ToolCorrectnessMetric(
            async_mode=False, should_exact_match=True
        )
        metric.measure(
            build_test_case(ToolCallType.MCP, ToolCallType.FUNCTION),
            _show_indicator=False,
        )

        assert metric.score == 0.0
        assert "Not an exact match" in metric.reason
        assert "Tool type mismatches" in metric.reason

    def test_type_mismatch_fails_with_ordering(self):
        metric = ToolCorrectnessMetric(
            async_mode=False, should_consider_ordering=True
        )
        metric.measure(
            build_test_case(ToolCallType.MCP, ToolCallType.FUNCTION),
            _show_indicator=False,
        )

        assert metric.score == 0.0
        assert "tool type mismatches" in metric.reason

    def test_defaulted_type_is_function(self):
        metric = ToolCorrectnessMetric(async_mode=False)
        test_case = LLMTestCase(
            input="What is the weather in Hong Kong?",
            actual_output="get_weather({})",
            tools_called=[ToolCall(name="get_weather")],
            expected_tools=[ToolCall(name="get_weather")],
        )
        metric.measure(test_case, _show_indicator=False)

        assert metric.score == 1.0

    def test_mixed_types_score_partially(self):
        metric = ToolCorrectnessMetric(async_mode=False)
        test_case = LLMTestCase(
            input="Weather in Tokyo and what the repo uses?",
            actual_output="get_weather({}) ask_question({})",
            tools_called=[
                ToolCall(name="get_weather", type=ToolCallType.FUNCTION),
                ToolCall(name="ask_question", type=ToolCallType.MCP),
            ],
            expected_tools=[
                ToolCall(name="get_weather", type=ToolCallType.FUNCTION),
                ToolCall(name="ask_question", type=ToolCallType.FUNCTION),
            ],
        )
        metric.measure(test_case, _show_indicator=False)

        assert metric.score == 0.5
        assert "ask_question (expected FUNCTION, called MCP)" in metric.reason

    def test_same_name_different_types_do_not_match(self):
        metric = ToolCorrectnessMetric(async_mode=False)
        test_case = LLMTestCase(
            input="Look up the repo twice",
            actual_output="ask_question({})",
            tools_called=[
                ToolCall(name="ask_question", type=ToolCallType.MCP),
                ToolCall(name="ask_question", type=ToolCallType.FUNCTION),
            ],
            expected_tools=[
                ToolCall(name="ask_question", type=ToolCallType.FUNCTION),
            ],
        )
        metric.measure(test_case, _show_indicator=False)

        assert metric.score == 1.0
