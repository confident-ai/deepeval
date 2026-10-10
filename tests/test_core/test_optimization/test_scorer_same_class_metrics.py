"""The optimizer scorer must average every configured metric instance.

Before the fix, ``Scorer._score_one`` and ``Scorer._a_score_one`` kept per-metric
scores in a dict keyed by ``metric.__class__.__name__``. Two metrics of the same
class (for example two ``GEval`` instances with different criteria) overwrote each
other, so only the last one listed counted and reordering the list changed which
prompt the optimizer selected.

These tests run offline: the generator and the metrics are stubs.
"""

import asyncio

import pytest

from deepeval.dataset.golden import Golden
from deepeval.metrics import BaseMetric
from deepeval.optimizer.scorer.scorer import Scorer
from deepeval.optimizer.types import PromptConfiguration
from deepeval.test_case import LLMTestCase


class FixedScoreMetric(BaseMetric):
    """A metric that returns a fixed score. Every instance shares this class."""

    def __init__(self, fixed_score: float, threshold: float = 0.5):
        self.fixed_score = fixed_score
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        self.score = self.fixed_score
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        self.score = self.fixed_score
        return self.score


class OtherFixedScoreMetric(FixedScoreMetric):
    """Same behaviour under a different class name, for the control case."""


def _scorer(metrics):
    # Bypass __init__: it validates a model callback and initialises an
    # optimizer model, neither of which the scoring path under test uses.
    scorer = Scorer.__new__(Scorer)
    scorer.metrics = list(metrics)
    scorer._semaphore = None
    scorer._throttle = 0.0
    scorer.generate = lambda prompts_by_module, golden: "actual output"

    async def a_generate(prompts_by_module, golden):
        return "actual output"

    scorer.a_generate = a_generate
    scorer._golden_to_test_case = lambda golden, actual: LLMTestCase(
        input=golden.input, actual_output=actual
    )
    return scorer


_GOLDEN = Golden(input="q", expected_output="a")
_CONFIG = PromptConfiguration.new(prompts={})


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 0.4],
        [0.4, 1.0],
    ],
)
def test_score_one_averages_every_same_class_metric(scores):
    scorer = _scorer([FixedScoreMetric(s) for s in scores])

    assert scorer._score_one(_CONFIG, _GOLDEN) == pytest.approx(0.7)


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 0.4],
        [0.4, 1.0],
    ],
)
def test_a_score_one_averages_every_same_class_metric(scores):
    scorer = _scorer([FixedScoreMetric(s) for s in scores])

    result = asyncio.run(scorer._a_score_one(_CONFIG, _GOLDEN))

    assert result == pytest.approx(0.7)


def test_metric_order_does_not_change_the_score():
    forward = _scorer([FixedScoreMetric(0.0), FixedScoreMetric(1.0)])
    backward = _scorer([FixedScoreMetric(1.0), FixedScoreMetric(0.0)])

    assert forward._score_one(_CONFIG, _GOLDEN) == pytest.approx(
        backward._score_one(_CONFIG, _GOLDEN)
    )


def test_distinct_classes_still_average():
    scorer = _scorer([FixedScoreMetric(1.0), OtherFixedScoreMetric(0.4)])

    assert scorer._score_one(_CONFIG, _GOLDEN) == pytest.approx(0.7)


def test_three_same_class_metrics_all_count():
    scorer = _scorer(
        [FixedScoreMetric(1.0), FixedScoreMetric(0.5), FixedScoreMetric(0.0)]
    )

    assert scorer._score_one(_CONFIG, _GOLDEN) == pytest.approx(0.5)
