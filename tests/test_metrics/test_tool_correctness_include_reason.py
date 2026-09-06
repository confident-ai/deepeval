import asyncio

import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class _StubLLM(DeepEvalBaseLLM):
    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs):
        raise AssertionError("LLM should not be called")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("LLM should not be called")

    def get_model_name(self, *args, **kwargs):
        return "stub-llm"


def _test_case() -> LLMTestCase:
    return LLMTestCase(
        input="Check the weather",
        actual_output="get_weather({})",
        tools_called=[ToolCall(name="get_weather")],
        expected_tools=[ToolCall(name="get_weather")],
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    ("include_reason", "expects_reason"),
    [(False, False), (True, True)],
)
def test_include_reason_controls_reason(
    async_mode: bool, include_reason: bool, expects_reason: bool
):
    metric = ToolCorrectnessMetric(
        model=_StubLLM(),
        async_mode=async_mode,
        include_reason=include_reason,
    )

    if async_mode:
        asyncio.run(metric.a_measure(_test_case(), _show_indicator=False))
    else:
        metric.measure(_test_case(), _show_indicator=False)

    assert metric.score == 1.0
    assert (metric.reason is not None) is expects_reason
    if expects_reason:
        assert "All expected tools ['get_weather'] were called" in metric.reason
