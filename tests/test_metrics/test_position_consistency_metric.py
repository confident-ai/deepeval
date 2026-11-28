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
    assert "all 2 judge calls" in metric.reason


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

    winner = metric.measure(arena("4 GOOD", "5"))

    # A 1-1 split is a tie: there is no winner to report.
    assert winner is None
    assert metric.winner is None
    assert metric.score == 0.5
    assert not metric.is_successful()
    assert metric.first_position_win_rate == 1.0
    assert "No contestant won a majority" in metric.reason


def test_every_position_and_pair_order_is_balanced():
    judge = FirstShownJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=judge,
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5", "22"))

    orders = [row["order"] for row in metric.winners_by_order]
    # Three contestants: rotations plus reversed rotations = all 6 orderings.
    assert len(winner_prompts(judge)) == 6
    assert len({tuple(order) for order in orders}) == 6
    for position in range(3):
        shown_here = [order[position] for order in orders]
        assert all(shown_here.count(name) == 2 for name in set(shown_here))
    names = ["model-0", "model-1", "model-2"]
    for a in names:
        for b in names:
            if a != b:
                before = sum(o.index(a) < o.index(b) for o in orders)
                assert before == 3
    assert metric.score == pytest.approx(1 / 3)
    assert metric.winner is None


def test_two_contestants_use_both_orders_only():
    judge = QualityJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=judge,
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5"))

    assert [row["order"] for row in metric.winners_by_order] == [
        ["model-0", "model-1"],
        ["model-1", "model-0"],
    ]


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


class AlternatingJudge(FakeJudge):
    """Picks the first contestant shown, then the last, then the first..."""

    def pick(self, contestants):
        calls = len([p for p in self.prompts if "Contestants:" in p])
        names = list(contestants)
        return names[0] if calls % 2 else names[-1]


@pytest.mark.parametrize("async_mode", [False, True])
def test_repeats_measure_judge_noise(async_mode):
    judge = QualityJudge()
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=judge,
        repeats=3,
        async_mode=async_mode,
    )

    metric.measure(arena("4 GOOD", "5"))

    assert len(winner_prompts(judge)) == 6
    assert metric.score == 1.0
    assert metric.repeat_consistency == 1.0
    assert "agreed 100%" in metric.reason


def test_noisy_judge_has_low_repeat_consistency():
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=AlternatingJudge(),
        repeats=2,
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5"))

    assert metric.repeat_consistency == 0.5


def test_repeat_consistency_is_none_without_repeats():
    metric = PositionConsistencyMetric(
        name="Correctness",
        evaluation_params=PARAMS,
        evaluation_steps=STEPS,
        model=QualityJudge(),
        async_mode=False,
    )

    metric.measure(arena("4 GOOD", "5"))

    assert metric.repeat_consistency is None
    assert "Repeats" not in metric.reason


def test_invalid_repeats_rejected():
    with pytest.raises(ValueError, match="repeats"):
        PositionConsistencyMetric(
            name="Correctness",
            evaluation_params=PARAMS,
            evaluation_steps=STEPS,
            model=QualityJudge(),
            repeats=0,
        )
