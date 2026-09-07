from typing import Optional, Union

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM


class _StubModel(DeepEvalBaseLLM):
    """The default (non-exact, unordered) scoring path and, with no
    ``available_tools``, the whole ``measure`` run never invoke the LLM. This
    model only needs to exist so the metric can be built without an API key."""

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-model"

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "duplicate-call scoring must not call the LLM; got prompt: %s"
            % prompt
        )

    async def a_generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "duplicate-call scoring must not call the LLM; got prompt: %s"
            % prompt
        )


def _measure(
    called: list, expected: list, *, evaluation_params=None
) -> ToolCorrectnessMetric:
    test_case = LLMTestCase(
        input="call the tools",
        actual_output="done",
        tools_called=called,
        expected_tools=expected,
    )
    metric = ToolCorrectnessMetric(
        async_mode=False,
        model=_StubModel(),
        evaluation_params=evaluation_params or [],
    )
    metric.measure(test_case, _show_indicator=False)
    return metric


def _search() -> ToolCall:
    return ToolCall(name="search")


class TestToolCorrectnessDuplicateCalls:
    """Regression tests for value-based dedup dropping repeated tool calls.

    ``ToolCall.__eq__``/``__hash__`` compare ``(name, input_parameters,
    output)`` only, so two genuinely distinct calls of the same tool compare
    equal. The non-exact scorer must therefore match expected calls against
    called calls *positionally* (one used call per expected call, even when the
    calls are value-identical) instead of deduplicating by value.
    """

    def test_two_identical_calls_both_expected_scores_one(self):
        # Regresses: used to score 0.5 because the second `search` was treated
        # as already consumed by the value-based dedup.
        metric = _measure([_search(), _search()], [_search(), _search()])
        assert metric.score == 1.0
        assert metric.success is True

    def test_only_one_of_two_expected_made_scores_half(self):
        # The correct down-scoring must be preserved: only one call happened.
        metric = _measure([_search()], [_search(), _search()])
        assert metric.score == 0.5

    def test_one_expected_twice_called_scores_one(self):
        metric = _measure([_search(), _search()], [_search()])
        assert metric.score == 1.0

    def test_distinct_tools_still_score_one(self):
        # Backward compatibility: distinct tools are unaffected.
        metric = _measure(
            [ToolCall(name="a"), ToolCall(name="b")],
            [ToolCall(name="a"), ToolCall(name="b")],
        )
        assert metric.score == 1.0

    def test_duplicates_share_matching_input_parameters(self):
        # Duplicate calls carrying the expected parameters match individually.
        with_params = ToolCall(name="search", input_parameters={"q": "x"})
        metric = _measure(
            [with_params, with_params],
            [with_params, with_params],
            evaluation_params=[],
        )
        assert metric.score == 1.0

    def test_empty_lists_score_one(self):
        metric = _measure([], [])
        assert metric.score == 1.0

    def test_missing_entirely_scores_zero(self):
        metric = _measure([], [_search()])
        assert metric.score == 0.0
