"""Tests for JudgeSelfConsistencyMetric.

The judge is scripted rather than real, so these run without an API key and
without any nondeterminism of their own. Each test hands the wrapped judge a
fixed sequence of scores and checks what the metric reports about them.
"""

import math
from typing import List, Optional, Union

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.metrics.community import JudgeSelfConsistencyMetric
from deepeval.metrics.community.judge_self_consistency.stats import (
    bootstrap_mean_interval,
    decision_flip_rate,
    stability,
)
from deepeval.metrics.utils import copy_metrics
from deepeval.test_case import LLMTestCase, SingleTurnParams

TEST_CASE = LLMTestCase(
    input="How tall is the Eiffel Tower?",
    actual_output="It is 330 metres tall.",
)


class ScriptedJudge(BaseMetric):
    """A judge that returns preset scores, one per call.

    The score list is shared by reference across the copies the metric makes
    for its replicates, so successive replicates consume successive scores
    regardless of the order they run in. An `Exception` in the list is raised
    instead of scored, which is how the error paths get exercised.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        scores: List[Union[float, Exception]],
        threshold: Optional[float] = 0.5,
        async_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.scores = scores
        self.threshold = threshold
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_model = "scripted-judge"
        self.using_native_model = False

    def measure(self, test_case, *args, **kwargs) -> float:
        self.error = None
        next_score = self.scores.pop(0)
        if isinstance(next_score, Exception):
            raise next_score
        self.score = next_score
        self.reason = f"scripted {next_score}"
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self):
        return "Scripted Judge"


class UndeclaredParamsJudge(BaseMetric):
    """A judge that never declares `_required_params` of its own."""

    def __init__(self, threshold: Optional[float] = 0.5):
        self.threshold = threshold
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_model = "scripted-judge"
        self.using_native_model = False

    def measure(self, test_case, *args, **kwargs) -> float:
        self.error = None
        self.score = 0.5
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self):
        return "Undeclared Judge"


# --- statistics -------------------------------------------------------------


def test_stability_is_one_when_every_repeat_agrees():
    assert stability([0.7, 0.7, 0.7, 0.7]) == 1.0


def test_stability_is_zero_at_maximum_spread():
    # Half at 0 and half at 1 is the widest a set of scores on [0, 1] can be.
    assert stability([0.0, 1.0, 0.0, 1.0]) == 0.0
    # An odd split reaches the floor too, unlike a population-variance reading.
    assert stability([0.0, 0.0, 1.0, 1.0, 1.0]) == 0.0


def test_stability_falls_as_scores_spread():
    tight = stability([0.50, 0.52, 0.48])
    loose = stability([0.20, 0.80, 0.50])
    assert 0.0 < loose < tight < 1.0


def test_stability_scales_with_the_declared_score_range():
    # The same repeats read as stable on a wide range and unstable on a
    # narrow one: spread only means something relative to the scale.
    assert stability([7.0, 8.0, 7.0, 8.0], (0.0, 20.0)) > 0.9
    assert stability([7.0, 8.0, 7.0, 8.0], (0.0, 10.0)) == pytest.approx(
        0.8845, abs=1e-4
    )
    assert stability([7.0, 8.0, 7.0, 8.0], (7.0, 8.0)) == 0.0


def test_stability_needs_two_scores():
    assert stability([0.4]) == 1.0
    assert stability([]) == 1.0


def test_flip_rate_is_zero_when_every_repeat_agrees():
    assert decision_flip_rate([True, True, True]) == 0.0
    assert decision_flip_rate([False, False, False]) == 0.0


def test_flip_rate_counts_disagreeing_pairs():
    # 2 passes and 2 fails => 4 disagreeing pairs out of 6 total pairs.
    assert decision_flip_rate([True, True, False, False]) == pytest.approx(
        4 / 6
    )
    # 1 pass and 4 fails => 4 disagreeing pairs out of 10.
    assert decision_flip_rate(
        [True, False, False, False, False]
    ) == pytest.approx(0.4)


def test_flip_rate_needs_two_decisions():
    assert decision_flip_rate([True]) is None
    assert decision_flip_rate([]) is None


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    scores = [0.2, 0.4, 0.6, 0.8, 0.5]
    mean = sum(scores) / len(scores)

    low, high = bootstrap_mean_interval(scores, resamples=500, seed=7)
    assert low <= mean <= high
    # Same seed, same interval.
    assert bootstrap_mean_interval(scores, resamples=500, seed=7) == (low, high)


def test_bootstrap_interval_collapses_when_every_score_is_equal():
    assert bootstrap_mean_interval([0.6, 0.6, 0.6], resamples=200, seed=1) == (
        pytest.approx(0.6),
        pytest.approx(0.6),
    )


def test_bootstrap_interval_rejects_nonsense_arguments():
    with pytest.raises(ValueError):
        bootstrap_mean_interval([0.1, 0.2], resamples=0)
    with pytest.raises(ValueError):
        bootstrap_mean_interval([0.1, 0.2], confidence=95)


# --- metric behaviour -------------------------------------------------------


def test_reports_perfect_stability_for_a_judge_that_agrees_with_itself():
    judge = ScriptedJudge(scores=[0.8] * 4)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=4, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.score == 1.0
    assert metric.decision_flip_rate == 0.0
    assert metric.is_successful() is True
    assert metric.flaky is False


def test_detects_a_judge_that_flips_its_own_verdict():
    # Four repeats hugging the judge's 0.5 threshold: two pass, two fail.
    judge = ScriptedJudge(scores=[0.49, 0.51, 0.49, 0.51], threshold=0.5)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=4, async_mode=False
    )

    metric.measure(TEST_CASE)

    # The scores barely move, so stability alone looks reassuring: it clears
    # the metric's own default threshold comfortably...
    assert metric.score > 0.95
    assert metric.is_successful() is True
    # ...while two thirds of repeat pairs reach the opposite verdict, because
    # the scores move across the judge's threshold rather than far.
    assert metric.decision_flip_rate == pytest.approx(4 / 6)
    assert "disagreed on pass/fail" in metric.reason
    # A verdict flip counts as disagreement however small the score movement.
    assert metric.flaky is True


def test_scores_low_when_repeats_spread_across_the_range():
    judge = ScriptedJudge(scores=[0.0, 1.0, 0.0, 1.0])
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=4, threshold=0.9, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.score == 0.0
    assert metric.is_successful() is False


def test_float_jitter_alone_is_not_disagreement():
    judge = ScriptedJudge(scores=[0.8, 0.8 + 1e-9], threshold=0.5)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=2, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.is_successful() is True
    assert metric.flaky is False


def test_auto_flaky_can_be_turned_off():
    judge = ScriptedJudge(scores=[0.49, 0.51, 0.49, 0.51], threshold=0.5)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=4, auto_flaky=False, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.flaky is False


def test_flaky_resets_between_runs():
    judge = ScriptedJudge(scores=[0.0, 1.0, 0.9, 0.9], threshold=0.5)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=2, async_mode=False
    )

    metric.measure(TEST_CASE)
    assert metric.flaky is True

    # A second, consistent run must not inherit the first run's verdict.
    metric.measure(TEST_CASE)
    assert metric.flaky is False


def test_measured_flaky_does_not_latch_into_copies():
    # `evaluate()` rebuilds metrics with `copy_metrics` for every test case, so
    # a flaky reading must not be mistaken for a caller-declared one.
    judge = ScriptedJudge(scores=[0.0, 1.0], threshold=0.5)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=2, async_mode=False
    )

    metric.measure(TEST_CASE)
    assert metric.flaky is True

    judge.scores.extend([0.7, 0.7])
    copied = copy_metrics([metric])[0]
    copied.measure(TEST_CASE)

    assert copied.flaky is False


def test_declared_flaky_survives_copying():
    judge = ScriptedJudge(scores=[0.7, 0.7], threshold=0.5)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=2, async_mode=False, flaky=True
    )

    copied = copy_metrics([metric])[0]
    copied.measure(TEST_CASE)

    assert copied.flaky is True


def test_reports_no_flip_rate_for_a_judge_without_a_threshold():
    judge = ScriptedJudge(scores=[0.3, 0.7, 0.5], threshold=None)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=3, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.decision_flip_rate is None
    assert "no threshold" in metric.reason


def test_records_the_spread_and_interval():
    judge = ScriptedJudge(scores=[0.2, 0.6, 0.4, 0.8])
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=4, async_mode=False
    )

    metric.measure(TEST_CASE)

    result = metric.result
    assert result.min_score == 0.2
    assert result.max_score == 0.8
    assert result.mean_score == pytest.approx(0.5)
    low, high = result.score_interval
    assert low <= result.mean_score <= high
    assert sorted(metric.replicate_scores) == [0.2, 0.4, 0.6, 0.8]


def test_honours_a_custom_score_range():
    judge = ScriptedJudge(scores=[7.0, 8.0, 7.0, 8.0], threshold=5.0)
    metric = JudgeSelfConsistencyMetric(
        metric=judge,
        replicates=4,
        score_range=(0.0, 10.0),
        async_mode=False,
    )

    metric.measure(TEST_CASE)

    assert metric.error is None
    assert metric.score > 0.85


def test_errors_on_scores_outside_the_declared_range():
    judge = ScriptedJudge(scores=[7.0, 8.0], threshold=5.0)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=2, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.error is not None
    assert "score_range" in metric.error
    assert metric.is_successful() is False


def test_tolerates_a_single_errored_replicate():
    judge = ScriptedJudge(scores=[0.5, RuntimeError("judge exploded"), 0.5])
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=3, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.error is None
    assert metric.score == 1.0
    assert metric.result.errored_replicates == 1
    assert "1 repeats errored" in metric.reason


def test_errors_when_too_few_replicates_survive():
    judge = ScriptedJudge(
        scores=[0.5, RuntimeError("judge exploded"), RuntimeError("again")]
    )
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=3, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.error is not None
    assert "judge exploded" in metric.error
    assert metric.is_successful() is False


def test_a_failed_run_does_not_report_the_previous_run_s_numbers():
    judge = ScriptedJudge(
        scores=[0.5, 0.5, RuntimeError("gone"), RuntimeError("gone")]
    )
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=2, async_mode=False
    )

    metric.measure(TEST_CASE)
    assert metric.score_breakdown is not None
    assert metric.verbose_logs is not None

    metric.measure(TEST_CASE)

    assert metric.error is not None
    assert metric.score is None
    assert metric.score_breakdown is None
    assert metric.verbose_logs is None


def test_missing_params_error_is_left_for_the_harness():
    # `MissingTestCaseParamsError` is how a test case gets skipped rather than
    # failed, so the wrapper must not swallow it into a replicate result.
    judge = ScriptedJudge(
        scores=[MissingTestCaseParamsError("no retrieval_context")] * 3
    )
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=3, async_mode=False
    )

    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(TEST_CASE)


def test_rejects_fewer_than_two_replicates():
    with pytest.raises(ValueError, match="at least 2"):
        JudgeSelfConsistencyMetric(
            metric=ScriptedJudge(scores=[0.5]), replicates=1
        )


def test_rejects_nonsense_constructor_arguments():
    judge = ScriptedJudge(scores=[0.5, 0.5])
    with pytest.raises(ValueError, match="confidence"):
        JudgeSelfConsistencyMetric(metric=judge, confidence=95)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        JudgeSelfConsistencyMetric(metric=judge, bootstrap_resamples=0)
    with pytest.raises(ValueError, match="score_range"):
        JudgeSelfConsistencyMetric(metric=judge, score_range=(1.0, 0.0))
    with pytest.raises(ValueError, match="max_concurrent"):
        JudgeSelfConsistencyMetric(metric=judge, max_concurrent=0)


def test_rejects_a_conversational_metric():
    class ConversationalJudge(BaseConversationalMetric):
        def measure(self, test_case, *args, **kwargs):
            return 1.0

        async def a_measure(self, test_case, *args, **kwargs):
            return 1.0

    with pytest.raises(TypeError, match="BaseMetric"):
        JudgeSelfConsistencyMetric(metric=ConversationalJudge())


def test_strict_mode_demands_perfect_stability():
    judge = ScriptedJudge(scores=[0.4, 0.6, 0.4, 0.6])
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=4, strict_mode=True, async_mode=False
    )

    metric.measure(TEST_CASE)

    assert metric.threshold == 1
    assert metric.score == 0
    assert metric.is_successful() is False


def test_inherits_the_required_params_of_the_wrapped_judge():
    judge = ScriptedJudge(scores=[0.5, 0.5])
    metric = JudgeSelfConsistencyMetric(metric=judge, replicates=2)

    assert metric._required_params == ScriptedJudge._required_params


def test_required_params_is_empty_when_the_judge_declares_none():
    # `BaseMetric` leaves `_required_params` as a bare type annotation, which
    # must not be mistaken for a real list of parameters.
    metric = JudgeSelfConsistencyMetric(
        metric=UndeclaredParamsJudge(), replicates=2
    )

    assert metric._required_params == []


def test_name_identifies_the_wrapped_judge():
    # Two wrappers in one run must not collapse into a single report row.
    relevancy = JudgeSelfConsistencyMetric(
        metric=ScriptedJudge(scores=[0.5, 0.5]), replicates=2
    )
    undeclared = JudgeSelfConsistencyMetric(
        metric=UndeclaredParamsJudge(), replicates=2
    )

    assert relevancy.__name__ == "Judge Self-Consistency (Scripted Judge)"
    assert undeclared.__name__ != relevancy.__name__


def test_async_measure_matches_sync():
    scores = [0.49, 0.51, 0.49, 0.51]
    sync_metric = JudgeSelfConsistencyMetric(
        metric=ScriptedJudge(scores=list(scores), threshold=0.5),
        replicates=4,
        async_mode=False,
    )
    async_metric = JudgeSelfConsistencyMetric(
        metric=ScriptedJudge(scores=list(scores), threshold=0.5),
        replicates=4,
        async_mode=True,
    )

    sync_metric.measure(TEST_CASE)
    async_metric.measure(TEST_CASE)

    assert math.isclose(sync_metric.score, async_metric.score)
    assert sync_metric.decision_flip_rate == async_metric.decision_flip_rate
    assert sync_metric.flaky == async_metric.flaky


def test_async_measure_respects_a_concurrency_bound():
    judge = ScriptedJudge(scores=[0.5] * 6, async_mode=True)
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=6, max_concurrent=2, async_mode=True
    )

    metric.measure(TEST_CASE)

    assert metric.score == 1.0
    assert len(metric.replicate_scores) == 6


def test_missing_params_error_propagates_from_the_async_path():
    judge = ScriptedJudge(
        scores=[MissingTestCaseParamsError("no retrieval_context")] * 3,
        async_mode=True,
    )
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=3, async_mode=True
    )

    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(TEST_CASE)


def test_async_path_folds_ordinary_replicate_errors_into_the_result():
    judge = ScriptedJudge(
        scores=[0.5, RuntimeError("judge exploded"), 0.5], async_mode=True
    )
    metric = JudgeSelfConsistencyMetric(
        metric=judge, replicates=3, async_mode=True
    )

    metric.measure(TEST_CASE)

    assert metric.error is None
    assert metric.result.errored_replicates == 1
