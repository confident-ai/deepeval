"""Tests for PositionConsistencyMetric.

The judges here are fakes, so no API key is needed. Each fake reads the
contestants from the prompt in the order they were shown and picks one:

- QualityJudge always picks the answer marked GOOD, wherever it appears
- FirstShownJudge always picks whichever contestant is shown first
"""

import json

import pytest

from deepeval.metrics.arena_g_eval.schema import Steps, Winner
from deepeval.metrics.community import PositionConsistencyMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)

PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
STEPS = ["Pick the answer that correctly answers the question."]


def shown_contestants(prompt: str) -> dict:
    """Contestants as rendered in the winner prompt, in display order."""
    block = prompt.split("Contestants:\n", 1)[1].split("\n\nParameters:", 1)[0]
    return json.loads(block)["arena_test_cases"]


class FakeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"

    def pick(self, contestants: dict) -> str:
        raise NotImplementedError

    def generate(self, prompt, *args, schema=None, **kwargs):
        self.prompts.append(prompt)
        if schema is Steps:
            return Steps(steps=STEPS)
        return Winner(winner=self.pick(shown_contestants(prompt)), reason="ok")

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        return self.generate(prompt, schema=schema)


class QualityJudge(FakeJudge):
    def pick(self, contestants):
        return next(
            name for name, body in contestants.items() if "GOOD" in body
        )


class FirstShownJudge(FakeJudge):
    def pick(self, contestants):
        return next(iter(contestants))


class BadNameJudge(FakeJudge):
    def pick(self, contestants):
        return "Nobody"


def arena(*outputs: str) -> ArenaTestCase:
    return ArenaTestCase(
        contestants=[
            Contestant(
                name=f"model-{i}",
                test_case=LLMTestCase(
                    input="What is 2 + 2?", actual_output=out
                ),
            )
            for i, out in enumerate(outputs)
        ]
    )


def winner_prompts(judge: FakeJudge) -> list:
    return [p for p in judge.prompts if "Contestants:" in p]


@pytest.mark.parametrize("async_mode", [False, True])
def test_consistent_judge_scores_one(async_mode):
    judge = QualityJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=judge,
        async_mode=async_mode,
    )

    winner = metric.measure(arena("4 GOOD", "5"))

    assert winner == "model-0"
    assert metric.score == 1.0
    assert metric.is_successful()
    assert metric.first_position_win_rate == 0.5
    assert "all 2 orderings" in metric.reason


@pytest.mark.parametrize("async_mode", [False, True])
def test_position_biased_judge_is_flagged(async_mode):
    judge = FirstShownJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=judge,
        async_mode=async_mode,
    )

    metric.measure(arena("4 GOOD", "5"))

    assert metric.score == 0.5
    assert not metric.is_successful()
    assert metric.first_position_win_rate == 1.0
    assert "1 of 2 orderings" in metric.reason


def test_every_contestant_is_shown_first_once():
    judge = FirstShownJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=judge,
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5", "22"))

    prompts = winner_prompts(judge)
    assert len(prompts) == 3
    first_shown = [next(iter(shown_contestants(p))) for p in prompts]
    assert len(set(first_shown)) == 3
    assert metric.score == pytest.approx(1 / 3)
    assert {row["winner"] for row in metric.winners_by_order} == {
        "model-0",
        "model-1",
        "model-2",
    }


def test_masking_is_fixed_across_orderings_and_hides_real_names():
    judge = QualityJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=judge,
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5"))

    prompts = winner_prompts(judge)
    masked = [set(shown_contestants(p)) for p in prompts]
    assert masked[0] == masked[1]
    assert all("model-" not in p for p in prompts)


def test_threshold_allows_partial_consistency():
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=FirstShownJudge(),
        threshold=0.5,
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5"))

    assert metric.is_successful()


def test_steps_are_generated_from_criteria():
    judge = QualityJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        criteria="Pick the correct answer.",
        model=judge,
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5"))

    assert metric.evaluation_steps == STEPS
    assert len(judge.prompts) == 3  # 1 steps prompt + 2 winner prompts


def test_unknown_winner_name_raises():
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=BadNameJudge(),
        async_mode=False,
    )

    with pytest.raises(ValueError, match="not one of the contestants"):
        metric.measure(arena("4 GOOD", "5"))


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        PositionConsistencyMetric(
            name="Correctness",
            evaluation_params=PARAMS,
            evaluation_steps=STEPS,
            model=QualityJudge(),
            threshold=1.5,
        )
