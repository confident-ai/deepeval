import json

import pytest
from pydantic import ValidationError

from deepeval.metrics import ConversationalGEval, GEval
from deepeval.metrics.conversational_g_eval.schema import (
    ReasonScore as ConversationalReasonScore,
)
from deepeval.metrics.g_eval.schema import ReasonScore as GEvalReasonScore
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MultiTurnParams,
    SingleTurnParams,
    Turn,
)


class _ScriptedScoreJudge(DeepEvalBaseLLM):
    def __init__(self, score: float):
        self.score = score

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        name = schema.__name__ if schema is not None else ""
        if name == "Steps":
            return json.dumps({"steps": ["Rate helpfulness."]})
        if name == "ReasonScore":
            return json.dumps({"score": self.score, "reason": "stub"})
        return json.dumps({"reason": "stub"})

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "scripted-judge"


@pytest.mark.parametrize(
    "schema_cls", [GEvalReasonScore, ConversationalReasonScore]
)
@pytest.mark.parametrize("bad_score", [-0.1, -5, 10.1, 25])
def test_reason_score_schema_rejects_out_of_bounds(schema_cls, bad_score):
    with pytest.raises(ValidationError):
        schema_cls(reason="stub", score=bad_score)


@pytest.mark.parametrize(
    "schema_cls", [GEvalReasonScore, ConversationalReasonScore]
)
@pytest.mark.parametrize("good_score", [0, 5.5, 10])
def test_reason_score_schema_accepts_bounds(schema_cls, good_score):
    parsed = schema_cls(reason="stub", score=good_score)
    assert parsed.score == good_score


@pytest.mark.parametrize(
    ("raw_score", "expected_score", "expect_clamped"),
    [
        (8, 0.8, False),
        (25, 1.0, True),
        (12.5, 1.0, True),
        (-5, 0.0, True),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
def test_geval_clamps_out_of_range_scores(
    raw_score, expected_score, expect_clamped, async_mode
):
    tc = LLMTestCase(input="hi", actual_output="Hello!")
    metric = GEval(
        name="helpfulness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Rate helpfulness.",
        model=_ScriptedScoreJudge(raw_score),
        async_mode=async_mode,
    )

    score = metric.measure(tc, _show_indicator=False)

    assert score == pytest.approx(expected_score)
    assert metric.score == pytest.approx(expected_score)
    if expect_clamped:
        assert "Score normalized:" in metric.verbose_logs
    else:
        assert "Score normalized:" not in metric.verbose_logs


@pytest.mark.parametrize(
    ("raw_score", "expected_score"),
    [(25, 1), (-5, 0), (1, 1), (0, 0)],
)
@pytest.mark.parametrize("async_mode", [False, True])
def test_geval_strict_mode_clamps_to_unit_interval(
    raw_score, expected_score, async_mode
):
    tc = LLMTestCase(input="hi", actual_output="Hello!")
    metric = GEval(
        name="helpfulness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Rate helpfulness.",
        model=_ScriptedScoreJudge(raw_score),
        strict_mode=True,
        async_mode=async_mode,
    )

    score = metric.measure(tc, _show_indicator=False)

    assert score == expected_score


def test_geval_custom_rubric_clamps_to_rubric_bounds():
    tc = LLMTestCase(input="hi", actual_output="Hello!")
    rubric = [
        Rubric(score_range=(2, 4), expected_outcome="Poor"),
        Rubric(score_range=(5, 8), expected_outcome="Great"),
    ]
    metric = GEval(
        name="helpfulness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Rate helpfulness.",
        rubric=rubric,
        model=_ScriptedScoreJudge(10),
        async_mode=False,
    )

    score = metric.measure(tc, _show_indicator=False)

    assert score == pytest.approx(1.0)
    assert "clamped to [2, 8]" in metric.verbose_logs


@pytest.mark.parametrize(
    ("raw_score", "expected_score", "expect_clamped"),
    [
        (8, 0.8, False),
        (25, 1.0, True),
        (-5, 0.0, True),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
def test_conversational_geval_clamps_out_of_range_scores(
    raw_score, expected_score, expect_clamped, async_mode
):
    tc = ConversationalTestCase(
        turns=[
            Turn(role="user", content="hi"),
            Turn(role="assistant", content="Hello!"),
        ]
    )
    metric = ConversationalGEval(
        name="helpfulness",
        evaluation_params=[
            MultiTurnParams.ROLE,
            MultiTurnParams.CONTENT,
        ],
        criteria="Rate helpfulness.",
        model=_ScriptedScoreJudge(raw_score),
        async_mode=async_mode,
    )

    score = metric.measure(tc, _show_indicator=False)

    assert score == pytest.approx(expected_score)
    assert metric.score == pytest.approx(expected_score)
    if expect_clamped:
        assert "Score normalized:" in metric.verbose_logs
    else:
        assert "Score normalized:" not in metric.verbose_logs
