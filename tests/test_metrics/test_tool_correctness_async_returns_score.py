import asyncio

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class _StubModel(DeepEvalBaseLLM):
    """ToolCorrectness never invokes the model when ``available_tools`` is
    unset, so the model only needs to exist for construction."""

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-model"

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "measure() must not call the LLM; got prompt: %s" % prompt
        )

    async def a_generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "measure() must not call the LLM; got prompt: %s" % prompt
        )


def _test_case() -> LLMTestCase:
    return LLMTestCase(
        input="query the web",
        actual_output="done",
        tools_called=[ToolCall(name="search")],
        expected_tools=[ToolCall(name="search")],
    )


def _metric(async_mode: bool = True) -> ToolCorrectnessMetric:
    return ToolCorrectnessMetric(
        model=_StubModel(),
        async_mode=async_mode,
        include_reason=False,
    )


class TestToolCorrectnessMeasureReturnsScore:
    def test_default_async_mode_measure_returns_score(self):
        # Regresses #3088: with async_mode left at its default (True),
        # measure() previously returned None even though self.score was set.
        metric = _metric()
        result = metric.measure(_test_case(), _show_indicator=False)
        assert metric.score == 1.0
        assert result == 1.0
        assert isinstance(result, float)

    def test_default_async_mode_is_async_mode(self):
        assert _metric().async_mode is True

    def test_sync_mode_measure_returns_score(self):
        metric = _metric(async_mode=False)
        result = metric.measure(_test_case(), _show_indicator=False)
        assert metric.score == 1.0
        assert result == 1.0

    def test_a_measure_returns_score(self):
        metric = _metric()
        result = asyncio.run(
            metric.a_measure(_test_case(), _show_indicator=False)
        )
        assert result == 1.0

    def test_returned_value_is_float_convertible(self):
        # The prompt optimizer does float(metric.measure(tc)); None breaks it.
        metric = _metric()
        result = metric.measure(_test_case(), _show_indicator=False)
        assert float(result) == 1.0

    def test_async_score_matches_sync_score(self):
        async_score = _metric().measure(_test_case(), _show_indicator=False)
        sync_score = _metric(async_mode=False).measure(
            _test_case(), _show_indicator=False
        )
        assert async_score == sync_score == 1.0

    def test_mismatch_scores_zero_and_returns_zero(self):
        tc = LLMTestCase(
            input="q",
            actual_output="a",
            tools_called=[ToolCall(name="search")],
            expected_tools=[ToolCall(name="browse")],
        )
        metric = _metric()
        result = metric.measure(tc, _show_indicator=False)
        assert metric.score == 0.0
        assert result == 0.0
