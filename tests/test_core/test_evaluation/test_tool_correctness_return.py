import asyncio

import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class NoCallModel(DeepEvalBaseLLM):
    def load_model(self):
        return None

    def generate(self, *args, **kwargs):
        raise AssertionError(
            "Deterministic tool matching must not call a judge"
        )

    async def a_generate(self, *args, **kwargs):
        raise AssertionError(
            "Deterministic tool matching must not call a judge"
        )

    def get_model_name(self):
        return "offline-tool-correctness"


@pytest.mark.parametrize("entrypoint", ["default", "sync", "async"])
@pytest.mark.parametrize(
    "called_names, strict_mode, expected_score",
    [
        pytest.param(["search", "lookup"], False, 1.0, id="full"),
        pytest.param(["search"], False, 0.5, id="partial"),
        pytest.param([], False, 0.0, id="zero"),
        pytest.param(["search"], True, 0.0, id="strict"),
    ],
)
def test_measurement_returns_score(
    entrypoint, called_names, strict_mode, expected_score
):
    test_case = LLMTestCase(
        input="Find the relevant information",
        actual_output="Done",
        tools_called=[ToolCall(name=name) for name in called_names],
        expected_tools=[ToolCall(name="search"), ToolCall(name="lookup")],
    )
    options = {"async_mode": False} if entrypoint == "sync" else {}
    metric = ToolCorrectnessMetric(
        model=NoCallModel(),
        eval_mode="llm",
        include_reason=False,
        strict_mode=strict_mode,
        **options,
    )

    if entrypoint == "async":
        result = asyncio.run(metric.a_measure(test_case, _show_indicator=False))
    else:
        result = metric.measure(test_case, _show_indicator=False)

    assert result == pytest.approx(expected_score)
    assert result == metric.score
    assert metric.success is (expected_score >= metric.threshold)
