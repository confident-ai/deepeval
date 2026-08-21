"""Deterministic tests for ConversationalGEval score normalization with rubrics.

These tests use a scripted fake judge model and do NOT require any API key.
They pin down that a `rubric` narrower than the default 0-10 scale is honoured:
  - the judge is asked for a score in the rubric's own range
  - the raw score is normalized against that range, not hard-coded /10
  - ConversationalGEval and GEval agree given the same rubric and raw score
  - the default (no rubric) behaviour is unchanged
"""

import pytest

import deepeval.metrics.conversational_g_eval.schema as cgschema
import deepeval.metrics.g_eval.schema as gschema
from deepeval.metrics import ConversationalGEval, GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MultiTurnParams,
    SingleTurnParams,
    Turn,
)


class ScriptedJudge(DeepEvalBaseLLM):
    """Always returns the same raw score and records the prompts it is sent."""

    def __init__(self, raw_score: int):
        self.raw_score = raw_score
        self.prompts = []
        super().__init__(model="scripted-judge")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        if schema in (cgschema.Steps, gschema.Steps):
            return schema(steps=["step one", "step two"])
        return schema(score=self.raw_score, reason="scripted reason")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "scripted-judge"


# A rubric that spans 0-5 rather than the default 0-10.
RUBRIC_0_TO_5 = [
    Rubric(score_range=(0, 1), expected_outcome="Unhelpful"),
    Rubric(score_range=(2, 3), expected_outcome="Partially helpful"),
    Rubric(score_range=(4, 5), expected_outcome="Fully helpful"),
]

# A rubric whose floor is not zero.
RUBRIC_2_TO_6 = [
    Rubric(score_range=(2, 3), expected_outcome="Unhelpful"),
    Rubric(score_range=(4, 6), expected_outcome="Helpful"),
]


def make_test_case() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What if these shoes don't fit?"),
            Turn(
                role="assistant",
                content="We offer a 30-day full refund at no extra cost.",
            ),
        ]
    )


def make_metric(judge: ScriptedJudge, rubric, async_mode: bool = False):
    return ConversationalGEval(
        name="Helpfulness",
        evaluation_params=[MultiTurnParams.CONTENT],
        criteria="Determine whether the assistant is helpful.",
        rubric=rubric,
        model=judge,
        async_mode=async_mode,
    )


@pytest.mark.parametrize(
    "raw_score,expected",
    [(0, 0.0), (3, 0.6), (5, 1.0)],
)
def test_rubric_range_is_used_to_normalize_score(raw_score, expected):
    judge = ScriptedJudge(raw_score)
    metric = make_metric(judge, RUBRIC_0_TO_5)
    metric.measure(make_test_case())
    assert metric.score == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw_score,expected",
    [(2, 0.0), (4, 0.5), (6, 1.0)],
)
def test_rubric_floor_is_subtracted_before_normalizing(raw_score, expected):
    judge = ScriptedJudge(raw_score)
    metric = make_metric(judge, RUBRIC_2_TO_6)
    metric.measure(make_test_case())
    assert metric.score == pytest.approx(expected)


def test_async_measure_normalizes_the_same_way():
    judge = ScriptedJudge(5)
    metric = make_metric(judge, RUBRIC_0_TO_5, async_mode=True)
    metric.measure(make_test_case())
    assert metric.score == pytest.approx(1.0)


@pytest.mark.parametrize("raw_score,expected", [(0, 0.0), (7, 0.7), (10, 1.0)])
def test_default_score_range_is_unchanged_without_a_rubric(raw_score, expected):
    judge = ScriptedJudge(raw_score)
    metric = make_metric(judge, None)
    metric.measure(make_test_case())
    assert metric.score == pytest.approx(expected)


def test_prompt_asks_for_a_score_in_the_rubric_range():
    judge = ScriptedJudge(5)
    metric = make_metric(judge, RUBRIC_0_TO_5)
    metric.measure(make_test_case())

    evaluation_prompt = judge.prompts[-1]
    assert "An integer from 0 to 5 (inclusive)" in evaluation_prompt
    assert "from 0 to 10" not in evaluation_prompt


def test_matches_geval_for_the_same_rubric_and_raw_score():
    conversational_judge = ScriptedJudge(5)
    conversational_metric = make_metric(conversational_judge, RUBRIC_0_TO_5)
    conversational_metric.measure(make_test_case())

    single_turn_judge = ScriptedJudge(5)
    single_turn_metric = GEval(
        name="Helpfulness",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        criteria="Determine whether the assistant is helpful.",
        rubric=RUBRIC_0_TO_5,
        model=single_turn_judge,
        async_mode=False,
    )
    single_turn_metric.measure(
        LLMTestCase(
            input="What if these shoes don't fit?",
            actual_output="We offer a 30-day full refund at no extra cost.",
        )
    )

    assert conversational_metric.score == pytest.approx(
        single_turn_metric.score
    )
