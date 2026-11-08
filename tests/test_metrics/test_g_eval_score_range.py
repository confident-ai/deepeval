import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
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


RUBRIC = [
    Rubric(score_range=(1, 3), expected_outcome="Poor."),
    Rubric(score_range=(4, 5), expected_outcome="Good."),
]


def _metric(score, rubric=None, async_mode=False):
    return GEval(
        name="Correctness",
        criteria="Is the output correct?",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=FixedScoreJudge(score),
        rubric=rubric,
        async_mode=async_mode,
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "score, rubric",
    [(15, None), (-3, None), (7, RUBRIC), (0, RUBRIC)],
)
def test_out_of_range_judge_score_raises(score, rubric, async_mode):
    metric = _metric(score, rubric=rubric, async_mode=async_mode)
    with pytest.raises(DeepEvalError, match="outside the expected range"):
        metric.measure(
            LLMTestCase(input="q", actual_output="a"), _show_indicator=False
        )


@pytest.mark.parametrize(
    "score, rubric, expected",
    [(0, None, 0.0), (10, None, 1.0), (7, None, 0.7), (5, RUBRIC, 1.0)],
)
def test_in_range_judge_score_is_normalized(score, rubric, expected):
    metric = _metric(score, rubric=rubric)
    assert metric.measure(
        LLMTestCase(input="q", actual_output="a"), _show_indicator=False
    ) == pytest.approx(expected)
