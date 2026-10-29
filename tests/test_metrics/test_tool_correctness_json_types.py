import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams


class OfflineModel(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def generate(self, *args, **kwargs):
        raise AssertionError(
            "Tool correctness must not call a model without available tools"
        )

    async def a_generate(self, *args, **kwargs):
        raise AssertionError(
            "Tool correctness must not call a model without available tools"
        )

    def get_model_name(self):
        return "offline"


def score(expected, called, field, *, exact=False, ordering=False):
    metric = ToolCorrectnessMetric(
        model=OfflineModel(),
        evaluation_params=[field],
        async_mode=False,
        should_exact_match=exact,
        should_consider_ordering=ordering,
    )
    kwargs = (
        "input_parameters"
        if field == ToolCallParams.INPUT_PARAMETERS
        else "output"
    )
    case = LLMTestCase(
        input="Run the tool",
        actual_output="done",
        expected_tools=[ToolCall(name="run", **{kwargs: expected})],
        tools_called=[ToolCall(name="run", **{kwargs: called})],
    )
    return metric.measure(case, _show_indicator=False)


@pytest.mark.parametrize(
    "exact,ordering", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize(
    "expected,called",
    [
        ({"enabled": True}, {"enabled": 1}),
        ({"enabled": False}, {"enabled": 0}),
        ({"config": {"enabled": True}}, {"config": {"enabled": 1}}),
        ({"steps": [{"enabled": True}]}, {"steps": [{"enabled": 1}]}),
    ],
)
def test_boolean_and_number_arguments_do_not_match(
    expected, called, exact, ordering
):
    assert (
        score(
            expected,
            called,
            ToolCallParams.INPUT_PARAMETERS,
            exact=exact,
            ordering=ordering,
        )
        == 0
    )


@pytest.mark.parametrize(
    "exact,ordering", [(False, False), (True, False), (False, True)]
)
def test_boolean_and_number_outputs_do_not_match(exact, ordering):
    assert (
        score(
            {"enabled": True},
            {"enabled": 1},
            ToolCallParams.OUTPUT,
            exact=exact,
            ordering=ordering,
        )
        == 0
    )


def test_matching_boolean_and_numeric_arguments_still_score_fully():
    assert (
        score(
            {"enabled": True},
            {"enabled": True},
            ToolCallParams.INPUT_PARAMETERS,
        )
        == 1
    )
    assert (
        score({"count": 1}, {"count": 1.0}, ToolCallParams.INPUT_PARAMETERS)
        == 1
    )


def test_partial_credit_ignores_a_boolean_number_mismatch():
    assert (
        score(
            {"config": {"enabled": True, "mode": "fast"}},
            {"config": {"enabled": 1, "mode": "fast"}},
            ToolCallParams.INPUT_PARAMETERS,
        )
        == 0.5
    )


@pytest.mark.asyncio
async def test_async_tool_comparison_distinguishes_booleans_from_numbers():
    metric = ToolCorrectnessMetric(
        model=OfflineModel(),
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )
    case = LLMTestCase(
        input="Enable logging",
        actual_output="done",
        expected_tools=[
            ToolCall(name="run", input_parameters={"enabled": True})
        ],
        tools_called=[ToolCall(name="run", input_parameters={"enabled": 1})],
    )
    assert await metric.a_measure(case, _show_indicator=False) == 0
