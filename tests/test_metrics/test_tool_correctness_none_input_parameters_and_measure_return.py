"""Regression tests for ToolCorrectnessMetric None-safety and measure() contract.

Two bugs are covered:

1. `_compare_dicts` crashed with `AttributeError: 'NoneType' object has no
   attribute 'keys'` when exactly one side of an input-parameters comparison
   was None (`ToolCall.input_parameters` defaults to None), breaking the
   non-exact-match and ordering code paths whenever
   `evaluation_params=[ToolCallParams.INPUT_PARAMETERS]`.
2. `measure()` returned None when `async_mode=True` (the constructor
   default): the async branch ran `a_measure` via `run_until_complete` but
   discarded its result, and only the sync branch had a `return`.

These tests run with a stub model, so they need no LLM provider API key.
"""

import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams


class _StubLLM(DeepEvalBaseLLM):
    """Minimal DeepEvalBaseLLM that never calls out to a provider."""

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs) -> str:
        raise AssertionError("generate should not be called in this test")

    async def a_generate(self, *args, **kwargs) -> str:
        raise AssertionError("a_generate should not be called in this test")

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-llm"


def _test_case(expected_tools, tools_called) -> LLMTestCase:
    return LLMTestCase(
        input="What is the weather in San Francisco?",
        actual_output="weather({})",
        expected_tools=expected_tools,
        tools_called=tools_called,
    )


class TestToolCorrectnessNoneInputParameters:
    """A missing input_parameters on one side must not crash _compare_dicts."""

    def test_measure_with_none_called_input_parameters_scores_zero(self):
        metric = ToolCorrectnessMetric(
            model=_StubLLM(),
            async_mode=False,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        test_case = _test_case(
            expected_tools=[
                ToolCall(name="weather", input_parameters={"city": "SF"})
            ],
            tools_called=[ToolCall(name="weather")],
        )

        metric.measure(test_case, _show_indicator=False)

        assert metric.score == 0.0

    def test_measure_with_both_input_parameters_none_scores_full(self):
        metric = ToolCorrectnessMetric(
            model=_StubLLM(),
            async_mode=False,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        test_case = _test_case(
            expected_tools=[ToolCall(name="weather")],
            tools_called=[ToolCall(name="weather")],
        )

        metric.measure(test_case, _show_indicator=False)

        assert metric.score == 1.0

    def test_measure_with_none_input_parameters_and_ordering_does_not_crash(
        self,
    ):
        metric = ToolCorrectnessMetric(
            model=_StubLLM(),
            async_mode=False,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
            should_consider_ordering=True,
        )
        test_case = _test_case(
            expected_tools=[
                ToolCall(name="weather", input_parameters={"city": "SF"})
            ],
            tools_called=[ToolCall(name="weather")],
        )

        metric.measure(test_case, _show_indicator=False)

        assert metric.score == 0.0


class TestToolCorrectnessMeasureReturnContract:
    """measure() must return the score in both async_mode and sync mode."""

    @pytest.mark.parametrize("async_mode", [True, False])
    def test_measure_returns_score_regardless_of_async_mode(self, async_mode):
        metric = ToolCorrectnessMetric(
            model=_StubLLM(),
            async_mode=async_mode,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        test_case = _test_case(
            expected_tools=[
                ToolCall(name="weather", input_parameters={"city": "SF"})
            ],
            tools_called=[
                ToolCall(name="weather", input_parameters={"city": "SF"})
            ],
        )

        returned_score = metric.measure(test_case, _show_indicator=False)

        assert returned_score is not None
        assert returned_score == metric.score
        assert metric.score == 1.0
