"""Tests for AnswerConsistencyMetric using a fake judge (no API key needed)."""

import pytest

from deepeval.metrics.community import AnswerConsistencyMetric
from deepeval.metrics.community.answer_consistency.schema import (
    AgreementScores,
    AgreementVerdict,
)
from deepeval.errors import MissingTestCaseParamsError
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


def verdict(**over):
    scores = dict(
        core_facts=5, specifics=5, supporting_detail=5, stance=5, format=5
    )
    scores.update(over)
    return AgreementVerdict(
        reason="judge reason",
        scores=AgreementScores(**scores),
    )


class FakeJudge(DeepEvalBaseLLM):
    """Scores a pair as disagreeing on `format` whenever a run containing OUTLIER is shown."""

    def __init__(self):
        self.prompts = []
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def _answer(self, prompt):
        self.prompts.append(prompt)
        return verdict(format=1) if "OUTLIER" in prompt else verdict()

    def generate(self, prompt, *args, schema=None, **kwargs):
        return self._answer(prompt)

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        return self._answer(prompt)

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"


def case(runs):
    return LLMTestCase(
        input="How long are gold backups retained?",
        actual_output=runs[0],
        metadata={"runs": runs},
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_identical_runs_are_fully_consistent_without_judge_calls(async_mode):
    judge = FakeJudge()
    metric = AnswerConsistencyMetric(model=judge, async_mode=async_mode)
    metric.measure(case(["30 days."] * 3 + ["30  DAYS."]))
    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert judge.prompts == []
    assert "fully agree" in metric.reason


@pytest.mark.parametrize("async_mode", [False, True])
def test_outlier_lowers_score_and_is_explained(async_mode):
    judge = FakeJudge()
    metric = AnswerConsistencyMetric(model=judge, async_mode=async_mode)
    metric.measure(
        case(["30 days."] * 3 + ["OUTLIER: thirty days, see policy"])
    )
    assert 0.0 < metric.score < 1.0
    assert metric.score_breakdown["format"] < 1.0
    assert metric.score_breakdown["core_facts"] == 1.0
    assert "format" in metric.reason and "judge reason" in metric.reason
    # the prompt carries the input and both responses
    assert any(
        "How long are gold backups retained?" in p for p in judge.prompts
    )


def test_out_of_range_judge_scores_are_clamped():
    class Wild(FakeJudge):
        def _answer(self, prompt):
            self.prompts.append(prompt)
            return verdict(format=9, core_facts=0)

    metric = AnswerConsistencyMetric(model=Wild(), async_mode=False)
    metric.measure(case(["a", "b"]))
    assert metric.score_breakdown["format"] == 1.0
    assert metric.score_breakdown["core_facts"] == 0.0


def test_response_text_cannot_rewrite_the_prompt():
    judge = FakeJudge()
    AnswerConsistencyMetric(model=judge, async_mode=False).measure(
        case(["contains {item_b}", "SECRET"])
    )
    assert judge.prompts[0].count("SECRET") == 1


def test_default_budget_is_one_judge_call_per_run():
    judge = FakeJudge()
    AnswerConsistencyMetric(model=judge, async_mode=False).measure(
        case(["a", "b", "c", "d", "e"])
    )
    assert len(judge.prompts) == 5  # not 10


def test_max_comparisons_and_strict_mode():
    judge = FakeJudge()
    metric = AnswerConsistencyMetric(
        model=judge, async_mode=False, max_comparisons=3, strict_mode=True
    )
    metric.measure(case(["a", "OUTLIER", "c", "d"]))
    assert len(judge.prompts) == 3
    assert metric.score in (0, 1.0)


def test_requires_multiple_runs():
    metric = AnswerConsistencyMetric(model=FakeJudge(), async_mode=False)
    with pytest.raises(MissingTestCaseParamsError, match="at least 2 runs"):
        metric.measure(case(["only one"]))
    with pytest.raises(MissingTestCaseParamsError, match="runs"):
        metric.measure(LLMTestCase(input="q", actual_output="a"))


def test_include_reason_false_has_no_reason():
    metric = AnswerConsistencyMetric(
        model=FakeJudge(), async_mode=False, include_reason=False
    )
    metric.measure(case(["a", "b"]))
    assert metric.reason is None
