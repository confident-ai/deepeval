"""Regression tests for ToolCorrectnessMetric with `input_parameters=None`.

`ToolCall.input_parameters` is optional, so comparing a tool called with
arguments against one called without them used to raise
`AttributeError: 'NoneType' object has no attribute 'keys'`.

No LLM call is made here: `available_tools` is not set, so the score is
computed entirely in code.
"""

import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams


@pytest.fixture(autouse=True)
def _dummy_openai_key(monkeypatch):
    # The metric builds its default judge model on init; it is never called.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _measure(called_params, expected_params, **metric_kwargs):
    test_case = LLMTestCase(
        input="What's the system status?",
        actual_output="Status checked.",
        tools_called=[
            ToolCall(name="check_status", input_parameters=called_params)
        ],
        expected_tools=[
            ToolCall(name="check_status", input_parameters=expected_params)
        ],
    )
    metric = ToolCorrectnessMetric(
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        async_mode=False,
        **metric_kwargs,
    )
    return metric.measure(test_case, _show_indicator=False)


@pytest.mark.parametrize("should_consider_ordering", [False, True])
@pytest.mark.parametrize(
    "called_params, expected_params, expected_score",
    [
        # Agent passes arguments to a tool that takes none.
        ({"verbose": True}, None, 0.0),
        # Agent omits the arguments the tool expects.
        (None, {"city": "San Francisco"}, 0.0),
        # `None` and `{}` both mean "no arguments".
        (None, {}, 1.0),
        ({}, None, 1.0),
        (None, None, 1.0),
    ],
)
def test_none_input_parameters_do_not_crash(
    called_params, expected_params, expected_score, should_consider_ordering
):
    score = _measure(
        called_params,
        expected_params,
        should_consider_ordering=should_consider_ordering,
    )
    assert score == expected_score


def test_partial_match_is_unchanged():
    score = _measure(
        {"city": "San Francisco", "unit": "C"},
        {"city": "San Francisco", "unit": "F"},
    )
    assert score == pytest.approx(0.5)
