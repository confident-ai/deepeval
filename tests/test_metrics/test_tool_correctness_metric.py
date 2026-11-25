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


def _measure(called, expected, **kwargs):
    metric = ToolCorrectnessMetric(async_mode=False, **kwargs)
    metric.measure(
        LLMTestCase(
            input="Book me a flight.",
            actual_output="Booked.",
            tools_called=called,
            expected_tools=expected,
        ),
        _show_indicator=False,
    )
    return metric


class TestToolCorrectnessMatching:
    def test_expected_tool_without_input_parameters_matches_any_call(self):
        metric = _measure(
            [ToolCall(name="search", input_parameters={"to": "NYC"})],
            [ToolCall(name="search")],
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
            should_consider_ordering=True,
        )
        assert metric.score == 1

    def test_expected_tool_without_output_matches_any_call(self):
        metric = _measure(
            [ToolCall(name="search", output=["UA1"])],
            [ToolCall(name="search")],
            evaluation_params=[ToolCallParams.OUTPUT],
            should_exact_match=True,
        )
        assert metric.score == 1

    def test_repeated_identical_calls_each_count(self):
        metric = _measure(
            [ToolCall(name="search"), ToolCall(name="search")],
            [ToolCall(name="search"), ToolCall(name="search")],
        )
        assert metric.score == 1

    def test_input_parameters_left_blank_are_not_compared(self):
        metric = _measure(
            [ToolCall(name="search", input_parameters={"to": "NYC"})],
            [ToolCall(name="search")],
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        assert metric.score == 1

    def test_call_without_input_parameters_does_not_match_expected_ones(self):
        metric = _measure(
            [ToolCall(name="search")],
            [ToolCall(name="search", input_parameters={"to": "NYC"})],
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        assert metric.score == 0
        assert "different input parameters for ['search']" in metric.reason

    def test_empty_expected_input_parameters_mean_none_expected(self):
        metric = _measure(
            [ToolCall(name="search", input_parameters={"to": "NYC"})],
            [ToolCall(name="search", input_parameters={})],
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        assert metric.score == 0


class TestToolCorrectnessReason:
    def test_reason_only_lists_tools_that_were_not_called(self):
        metric = _measure(
            [
                ToolCall(
                    name="item_not_arrived_flow",
                    input_parameters={"tracking": "MP867"},
                    output={"status": "lost"},
                )
            ],
            [
                ToolCall(name="item_not_arrived_flow"),
                ToolCall(name="cancel_order"),
            ],
        )
        assert metric.score == 0.5
        assert "missing tools ['cancel_order']" in metric.reason

    def test_reason_names_tools_with_different_input_parameters(self):
        metric = _measure(
            [
                ToolCall(name="search", input_parameters={"to": "LAX"}),
                ToolCall(name="book"),
            ],
            [
                ToolCall(name="search", input_parameters={"to": "NYC"}),
                ToolCall(name="book"),
            ],
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        assert metric.score == 0.5
        assert "different input parameters for ['search']" in metric.reason
        assert "missing tools" not in metric.reason

    def test_reason_names_tools_with_a_different_output(self):
        metric = _measure(
            [ToolCall(name="search", output=["UA2"])],
            [ToolCall(name="search", output=["UA1"])],
            evaluation_params=[ToolCallParams.OUTPUT],
        )
        assert metric.score == 0
        assert "different output for ['search']" in metric.reason

    def test_ordering_does_not_call_a_missing_tool_out_of_order(self):
        metric = _measure(
            [ToolCall(name="search")],
            [ToolCall(name="search"), ToolCall(name="book")],
            should_consider_ordering=True,
        )
        assert "missing tools ['book']" in metric.reason
        assert "out-of-order" not in metric.reason

    def test_ordering_names_tools_with_different_input_parameters(self):
        metric = _measure(
            [ToolCall(name="search", input_parameters={"to": "NYC", "on": 1})],
            [ToolCall(name="search", input_parameters={"to": "NYC", "on": 2})],
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
            should_consider_ordering=True,
        )
        assert metric.score == 0.5
        assert "different input parameters for ['search']" in metric.reason

    def test_reason_says_when_no_tools_were_expected(self):
        metric = _measure([ToolCall(name="search")], [])
        assert metric.score == 0
        assert "no tools were expected, but ['search'] were called" in (
            metric.reason
        )

    def test_include_reason_false_leaves_no_reason(self):
        metric = _measure(
            [ToolCall(name="search")],
            [ToolCall(name="search")],
            include_reason=False,
        )
        assert metric.reason is None
