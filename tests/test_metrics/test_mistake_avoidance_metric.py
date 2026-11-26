"""Tests for MistakeAvoidanceMetric.

These tests use a fake DeepEvalBaseLLM judge so they run without any API key.
They prove the metric scores the fraction of known mistakes an answer avoids,
fails an answer that repeats known mistakes, and passes one that avoids them.
"""

import json

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.base_metric import Verdict
from deepeval.metrics.community import MistakeAvoidanceMetric
from deepeval.metrics.community.mistake_avoidance.schema import (
    MistakeAvoidanceVerdict,
    Verdicts,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class FakeJudge(DeepEvalBaseLLM):
    """Returns a preset reply, capturing the prompt it was given."""

    def __init__(self, reply):
        self._reply = reply
        self.last_prompt = None
        self.calls = 0
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        self.calls += 1
        return self._reply

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        self.calls += 1
        return self._reply

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"


def verdicts(*items):
    return Verdicts(
        verdicts=[
            MistakeAvoidanceVerdict(verdict=verdict, reason=reason)
            for verdict, reason in items
        ]
    )


QUERY = "What is the capital of Australia, and how many people live there?"
KNOWN_MISTAKES = [
    'Answered "Sydney", the largest city; the capital is Canberra.',
    "Gave the population of the metropolitan area instead of the city proper.",
]


def make_test_case(actual_output):
    return LLMTestCase(
        input=QUERY,
        actual_output=actual_output,
        context=KNOWN_MISTAKES,
    )


def test_fails_when_every_known_mistake_is_repeated():
    judge = FakeJudge(
        verdicts(
            (Verdict.NO, "The answer names Sydney as the capital."),
            (Verdict.NO, "The figure given is for the metropolitan area."),
        )
    )
    metric = MistakeAvoidanceMetric(model=judge, async_mode=False)

    metric.measure(
        make_test_case("The capital is Sydney, home to 5.3 million people.")
    )

    assert metric.score == 0.0
    assert metric.is_successful() is False
    assert "avoided 0 of 2 known mistakes" in metric.reason
    assert "The answer names Sydney as the capital." in metric.reason
    # The prompt must show the judge every known mistake, numbered.
    assert '1. Answered "Sydney"' in judge.last_prompt
    assert "2. Gave the population" in judge.last_prompt
    assert "exactly 2 objects" in judge.last_prompt


def test_passes_when_every_known_mistake_is_avoided():
    judge = FakeJudge(
        verdicts(
            (Verdict.YES, "The answer names Canberra."),
            (Verdict.YES, "The figure given is for the city proper."),
        )
    )
    metric = MistakeAvoidanceMetric(model=judge, async_mode=False)

    metric.measure(
        make_test_case("The capital is Canberra, home to about 470,000 people.")
    )

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert metric.reason == "The answer avoided 2 of 2 known mistakes."


def test_score_is_the_fraction_of_mistakes_avoided():
    judge = FakeJudge(
        verdicts(
            (Verdict.YES, "The answer names Canberra."),
            (Verdict.NO, "The figure given is for the metropolitan area."),
        )
    )
    metric = MistakeAvoidanceMetric(model=judge, async_mode=False)
    test_case = make_test_case(
        "The capital is Canberra, home to 5.3 million people."
    )

    metric.measure(test_case)

    assert metric.score == 0.5
    # 0.5 meets the default threshold of 0.5.
    assert metric.is_successful() is True
    assert "avoided 1 of 2 known mistakes" in metric.reason

    stricter = MistakeAvoidanceMetric(
        model=judge, threshold=1.0, async_mode=False
    )
    stricter.measure(test_case)
    assert stricter.score == 0.5
    assert stricter.is_successful() is False


def test_strict_mode_scores_zero_unless_every_mistake_is_avoided():
    judge = FakeJudge(
        verdicts(
            (Verdict.YES, "The answer names Canberra."),
            (Verdict.NO, "The figure given is for the metropolitan area."),
        )
    )
    metric = MistakeAvoidanceMetric(
        model=judge, strict_mode=True, async_mode=False
    )

    metric.measure(
        make_test_case("The capital is Canberra, home to 5.3 million people.")
    )

    assert metric.threshold == 1
    assert metric.score == 0
    assert metric.is_successful() is False


def test_async_measure_matches_sync():
    judge = FakeJudge(
        verdicts(
            (Verdict.YES, "The answer names Canberra."),
            (Verdict.NO, "The figure given is for the metropolitan area."),
        )
    )
    metric = MistakeAvoidanceMetric(model=judge, async_mode=True)

    metric.measure(
        make_test_case("The capital is Canberra, home to 5.3 million people.")
    )

    assert metric.score == 0.5
    assert "avoided 1 of 2 known mistakes" in metric.reason


def test_reason_costs_no_extra_judge_call_and_can_be_turned_off():
    judge = FakeJudge(
        verdicts((Verdict.NO, "Names Sydney."), (Verdict.YES, "ok"))
    )
    metric = MistakeAvoidanceMetric(
        model=judge, include_reason=False, async_mode=False
    )

    metric.measure(make_test_case("The capital is Sydney."))

    assert metric.reason is None
    assert judge.calls == 1

    with_reason = MistakeAvoidanceMetric(model=judge, async_mode=False)
    with_reason.measure(make_test_case("The capital is Sydney."))
    assert with_reason.reason is not None
    assert judge.calls == 2


def test_accepts_a_plain_json_reply_from_the_judge():
    # A custom model that ignores the schema replies with a JSON string.
    judge = FakeJudge(
        json.dumps(
            {
                "verdicts": [
                    {"verdict": "yes", "reason": "The answer names Canberra."},
                    {"verdict": "no", "reason": "Metropolitan-area figure."},
                ]
            }
        )
    )
    metric = MistakeAvoidanceMetric(model=judge, async_mode=False)

    metric.measure(
        make_test_case("The capital is Canberra, home to 5.3 million people.")
    )

    assert metric.score == 0.5


def test_requires_known_mistakes_in_context():
    judge = FakeJudge(verdicts())
    metric = MistakeAvoidanceMetric(model=judge, async_mode=False)

    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(
            LLMTestCase(input=QUERY, actual_output="The capital is Canberra.")
        )
    assert judge.calls == 0
