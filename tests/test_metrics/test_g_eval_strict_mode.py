import json

import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.schema import ReasonScore, Steps
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams


class FixedScoreJudge(DeepEvalBaseLLM):
    def __init__(self, score):
        self.score = score
        super().__init__(model="fixed-score-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        if schema is Steps:
            return Steps(steps=["Check the output."])
        return ReasonScore(reason="Scored.", score=self.score)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "fixed-score-judge"


class JsonScoreJudge(FixedScoreJudge):
    """Returns raw JSON, so the score skips ReasonScore validation."""

    def generate(self, prompt, schema=None, **kwargs):
        return json.dumps(
            {"steps": ["Check the output."], "score": self.score, "reason": "r"}
        )


def _strict_metric(score, async_mode=False, judge_cls=FixedScoreJudge):
    return GEval(
        name="Correctness",
        criteria="Is the output correct?",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=judge_cls(score),
        strict_mode=True,
        async_mode=async_mode,
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("score", [5, 10, 1.7, 0.9, -1])
def test_strict_mode_rejects_scores_other_than_0_or_1(score, async_mode):
    metric = _strict_metric(score, async_mode=async_mode)
    with pytest.raises(DeepEvalError, match="only allows 0 or 1"):
        metric.measure(
            LLMTestCase(input="q", actual_output="a"), _show_indicator=False
        )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "score, expected_success",
    [(1, True), (0, False), (1.0, True), (0.0, False)],
)
def test_strict_mode_keeps_0_and_1(score, expected_success, async_mode):
    metric = _strict_metric(score, async_mode=async_mode)
    result = metric.measure(
        LLMTestCase(input="q", actual_output="a"), _show_indicator=False
    )
    assert result == int(score)
    assert type(metric.score) is int
    assert metric.success is expected_success


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "score", [None, "abc", "0.99999999999999999", "5", [1], 10**400]
)
def test_strict_mode_rejects_malformed_json_scores(score, async_mode):
    metric = _strict_metric(
        score, async_mode=async_mode, judge_cls=JsonScoreJudge
    )
    with pytest.raises(DeepEvalError, match="only allows 0 or 1"):
        metric.measure(
            LLMTestCase(input="q", actual_output="a"), _show_indicator=False
        )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "score, expected",
    [(1, 1), (0, 0), ("1", 1), ("0", 0), ("1.0", 1), ("0.0", 0), (1.0, 1)],
)
def test_strict_mode_keeps_0_and_1_from_json(score, expected, async_mode):
    metric = _strict_metric(
        score, async_mode=async_mode, judge_cls=JsonScoreJudge
    )
    metric.measure(
        LLMTestCase(input="q", actual_output="a"), _show_indicator=False
    )
    assert metric.score == expected
    assert type(metric.score) is int
    assert metric.success is (expected == 1)
