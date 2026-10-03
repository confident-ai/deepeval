import os
import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import (
    LLMTestCase,
    ToolCall,
    ToolCallParams,
    ToolCallType,
)

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


def _book(**params):
    return ToolCall(name="book", input_parameters=params)


def _measure(expected, called):
    metric = ToolCorrectnessMetric(
        async_mode=False,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )
    metric.measure(
        LLMTestCase(
            input="Book two rooms",
            actual_output="book(); book()",
            tools_called=called,
            expected_tools=expected,
        ),
        _show_indicator=False,
    )
    return metric.score


class TestToolCorrectnessMetricAssignment:
    """In the default (unordered) mode each expected call is paired with at
    most one actual call. The pairing must be the best available one, so the
    score cannot depend on the order expected_tools happen to be listed in."""

    EXPECTED = [_book(a=1, b=1, c=1), _book(a=1, b=1, c=2)]
    CALLED = [_book(a=1, b=1, c=2), _book(a=1, b=5, c=5)]

    def test_score_does_not_depend_on_expected_order(self):
        forward = _measure(self.EXPECTED, self.CALLED)
        reverse = _measure(self.EXPECTED[::-1], self.CALLED)
        assert forward == pytest.approx(reverse)

    def test_partial_credit_uses_best_pairing(self):
        # Best pairing: expected[1] -> called[0] (3/3), expected[0] ->
        # called[1] (1/3). Greedy took expected[0] -> called[0] (2/3) and left
        # expected[1] with called[1] (1/3), scoring 0.5.
        assert _measure(self.EXPECTED, self.CALLED) == pytest.approx(2 / 3)

    def test_score_does_not_depend_on_called_order(self):
        forward = _measure(self.EXPECTED, self.CALLED)
        reverse = _measure(self.EXPECTED, self.CALLED[::-1])
        assert forward == pytest.approx(reverse)

    def test_more_calls_than_expected(self):
        called = self.CALLED + [_book(a=1, b=1, c=1)]
        assert _measure(self.EXPECTED, called) == pytest.approx(1.0)

    def test_fewer_calls_than_expected(self):
        assert _measure(self.EXPECTED, [_book(a=1, b=1, c=2)]) == pytest.approx(
            0.5
        )
