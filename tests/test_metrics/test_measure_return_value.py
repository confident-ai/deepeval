import os

os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-regression-test")

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall


def build_test_case():
    return LLMTestCase(
        input="q",
        actual_output="ask_question({})",
        tools_called=[ToolCall(name="ask_question")],
        expected_tools=[ToolCall(name="ask_question")],
    )


class TestMeasureReturnValue:
    """measure() must return the score on the default async path (#3088).

    The async_mode branch ran a_measure() without returning it, so the
    default path handed back None while metric.score was still set — and
    callers like the prompt optimizer's _measure_no_indicator (float(score))
    raised TypeError.
    """

    def test_measure_returns_score_with_default_async_mode(self):
        metric = ToolCorrectnessMetric()
        assert metric.async_mode is True  # the default is the broken path
        score = metric.measure(build_test_case(), _show_indicator=False)
        assert isinstance(score, float)
        assert score == metric.score == 1.0

    def test_optimizer_scorer_helper_receives_float(self):
        from deepeval.optimizer.scorer.utils import _measure_no_indicator

        score = _measure_no_indicator(
            ToolCorrectnessMetric(), build_test_case()
        )
        assert isinstance(score, float)

    def test_sync_mode_still_matches_async_result(self):
        sync_metric = ToolCorrectnessMetric(async_mode=False)
        async_metric = ToolCorrectnessMetric()
        assert sync_metric.measure(
            build_test_case(), _show_indicator=False
        ) == async_metric.measure(build_test_case(), _show_indicator=False)
