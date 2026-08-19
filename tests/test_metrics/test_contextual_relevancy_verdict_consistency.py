"""Regression tests for ContextualRelevancyMetric verdict classification.

`_calculate_score` and the reason-generation helpers (`_generate_reason` /
`_a_generate_reason`) used to classify verdicts with two different
predicates: the score path only treated an exact `"yes"` as relevant, while
the reason path only treated an exact `"no"` as irrelevant (everything else
fell into the "relevant" `else` branch). A non-canonical judge response such
as `"yes."`, `"yes "`, or `"yes<stray text>"` is therefore neither an exact
`"yes"` nor an exact `"no"`, so it is scored as irrelevant while the reason
text describes it as relevant, producing contradictory output.

Both paths now share `_verdict_is_relevant`, so a verdict can only ever be
counted on one side of the classification. These tests drive the metric
through `measure()`/`a_measure()` with a stub judge (no network, no API
key) and assert that the score and the statements handed to the reason
prompt always agree.
"""

import asyncio

import pytest

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy.contextual_relevancy import (
    _verdict_is_relevant,
)
from deepeval.metrics.contextual_relevancy.schema import (
    ContextualRelevancyScoreReason,
    ContextualRelevancyVerdicts,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class _StubJudge(DeepEvalBaseLLM):
    """Always returns the same verdict token for every statement, and
    records whether the reason-generation prompt treated the statements
    as relevant or irrelevant."""

    def __init__(self, verdict_token: str):
        self.verdict_token = verdict_token
        self.captured_reason_prompts = []

    def load_model(self, *args, **kwargs):
        return self

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-judge"

    def generate(self, prompt, schema=None, **kwargs):
        if schema is ContextualRelevancyVerdicts:
            return ContextualRelevancyVerdicts(
                verdicts=[
                    {
                        "statement": f"statement {i}",
                        "verdict": self.verdict_token,
                        "reason": None,
                    }
                    for i in range(2)
                ]
            )
        self.captured_reason_prompts.append(str(prompt))
        return ContextualRelevancyScoreReason(reason="stub reason")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)


@pytest.mark.parametrize(
    "verdict_token",
    ["yes", "yes ", "yes.", "yesշո", "Yes"],
)
def test_non_canonical_yes_verdicts_score_and_reason_agree_sync(
    verdict_token: str,
):
    judge = _StubJudge(verdict_token)
    metric = ContextualRelevancyMetric(
        model=judge, async_mode=False, include_reason=True
    )
    test_case = LLMTestCase(
        input="what is x?",
        actual_output="x is y",
        retrieval_context=["some context"],
    )

    metric.measure(test_case, _show_indicator=False)

    assert metric.score == 1.0
    reason_prompt = judge.captured_reason_prompts[0]
    assert "statement 0" in reason_prompt
    assert "statement 1" in reason_prompt


@pytest.mark.parametrize(
    "verdict_token",
    ["yes", "yes ", "yes.", "yesշո"],
)
def test_non_canonical_yes_verdicts_score_and_reason_agree_async(
    verdict_token: str,
):
    judge = _StubJudge(verdict_token)
    metric = ContextualRelevancyMetric(
        model=judge, async_mode=True, include_reason=True
    )
    test_case = LLMTestCase(
        input="what is x?",
        actual_output="x is y",
        retrieval_context=["some context"],
    )

    asyncio.run(metric.a_measure(test_case, _show_indicator=False))

    assert metric.score == 1.0
    reason_prompt = judge.captured_reason_prompts[0]
    assert "statement 0" in reason_prompt
    assert "statement 1" in reason_prompt


def test_no_verdict_is_still_irrelevant():
    judge = _StubJudge("no")
    metric = ContextualRelevancyMetric(
        model=judge, async_mode=False, include_reason=True
    )
    test_case = LLMTestCase(
        input="what is x?",
        actual_output="x is y",
        retrieval_context=["some context"],
    )

    metric.measure(test_case, _show_indicator=False)

    assert metric.score == 0.0
    reason_prompt = judge.captured_reason_prompts[0]
    assert "statement 0" not in reason_prompt
    assert "statement 1" not in reason_prompt


@pytest.mark.parametrize(
    "verdict,expected",
    [
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        (" yes ", True),
        ("yes.", True),
        ("yesշո", True),
        ("no", False),
        ("No", False),
        (" no ", False),
        ("no.", False),
        ("", False),
    ],
)
def test_verdict_is_relevant_helper(verdict: str, expected: bool):
    assert _verdict_is_relevant(verdict) is expected
