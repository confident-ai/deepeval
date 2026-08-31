"""Offline tests for ToolCorrectnessMetric robustness to missing tool arguments.

``ToolCall.input_parameters`` defaults to ``None`` (tools commonly take no
arguments). When ``INPUT_PARAMETERS`` is enabled for scoring, comparing a tool
with no argument list against one with arguments crashed on ``None.keys()`` in
``_compare_dicts``. These tests pin the new, graceful behavior and guard the
default scoring paths against regression.

The whole metric is exercised offline: a stub ``DeepEvalBaseLLM`` is used, and
tool-selection scoring is skipped (no ``available_tools``), so no LLM is called.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    LLMTestCase,
    ToolCall,
    ToolCallParams,
    ToolCallType,
)


def _stub_model() -> DeepEvalBaseLLM:
    m = MagicMock(spec=DeepEvalBaseLLM)
    m.get_model_name.return_value = "mock-llm"
    m.supports_multimodal.return_value = False
    return m


def _make_metric(
    *, should_consider_ordering: bool = False, evaluation_params=None
) -> ToolCorrectnessMetric:
    return ToolCorrectnessMetric(
        model=_stub_model(),
        async_mode=False,
        should_consider_ordering=should_consider_ordering,
        evaluation_params=evaluation_params or [],
    )


def _case(tools_called, expected_tools) -> LLMTestCase:
    return LLMTestCase(
        input="q",
        tools_called=tools_called,
        expected_tools=expected_tools,
    )


class TestCompareDictsNoneGuard:
    def test_none_vs_dict_no_crash(self):
        m = _make_metric()
        assert m._compare_dicts(None, {"city": "SF"}) == 0.0

    def test_dict_vs_none_no_crash(self):
        m = _make_metric()
        assert m._compare_dicts({"city": "SF"}, None) == 0.0

    def test_both_none_is_equal(self):
        m = _make_metric()
        assert m._compare_dicts(None, None) == 1.0

    def test_equal_dicts_unchanged(self):
        m = _make_metric()
        assert m._compare_dicts({"city": "SF"}, {"city": "SF"}) == 1.0


class TestMeasureNoCrash:
    def test_expected_without_args_called_with_args(self):
        """expected='get_weather' (no args) vs called='get_weather'({'city': ...})."""
        metric = _make_metric(
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        )
        score = metric.measure(
            _case(
                tools_called=[
                    ToolCall(
                        name="get_weather",
                        type=ToolCallType.FUNCTION,
                        input_parameters={"city": "SF"},
                    )
                ],
                expected_tools=[
                    ToolCall(name="get_weather", type=ToolCallType.FUNCTION)
                ],
            )
        )
        # Args differ on only side -> argument mismatch -> 0
        assert score == 0.0
        assert metric.reason is not None

    def test_expected_with_args_called_without_args(self):
        metric = _make_metric(
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        )
        score = metric.measure(
            _case(
                tools_called=[
                    ToolCall(name="get_weather", type=ToolCallType.FUNCTION)
                ],
                expected_tools=[
                    ToolCall(
                        name="get_weather",
                        type=ToolCallType.FUNCTION,
                        input_parameters={"city": "SF"},
                    )
                ],
            )
        )
        assert score == 0.0

    def test_ordering_path_no_crash(self):
        metric = _make_metric(
            should_consider_ordering=True,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        score = metric.measure(
            _case(
                tools_called=[
                    ToolCall(
                        name="get_weather",
                        type=ToolCallType.FUNCTION,
                        input_parameters={"city": "SF"},
                    )
                ],
                expected_tools=[
                    ToolCall(name="get_weather", type=ToolCallType.FUNCTION)
                ],
            )
        )
        assert score == 0.0


class TestDefaultBehaviorRegression:
    def test_matching_args_score_one(self):
        metric = _make_metric(
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        )
        score = metric.measure(
            _case(
                tools_called=[
                    ToolCall(
                        name="get_weather",
                        type=ToolCallType.FUNCTION,
                        input_parameters={"city": "SF"},
                    )
                ],
                expected_tools=[
                    ToolCall(
                        name="get_weather",
                        type=ToolCallType.FUNCTION,
                        input_parameters={"city": "SF"},
                    )
                ],
            )
        )
        assert score == 1.0

    def test_default_params_ignore_args(self):
        """Without INPUT_PARAMETERS, argument lists are not compared (unchanged)."""
        metric = _make_metric()
        score = metric.measure(
            _case(
                tools_called=[
                    ToolCall(name="get_weather", type=ToolCallType.FUNCTION)
                ],
                expected_tools=[
                    ToolCall(name="get_weather", type=ToolCallType.FUNCTION)
                ],
            )
        )
        assert score == 1.0

    def test_call_with_args_but_expected_without_args_sans_param_check_ok(self):
        """Default behaviour: args ignored unless INPUT_PARAMETERS requested."""
        metric = _make_metric()
        score = metric.measure(
            _case(
                tools_called=[
                    ToolCall(
                        name="get_weather",
                        type=ToolCallType.FUNCTION,
                        input_parameters={"city": "SF"},
                    )
                ],
                expected_tools=[
                    ToolCall(name="get_weather", type=ToolCallType.FUNCTION)
                ],
            )
        )
        assert score == 1.0
