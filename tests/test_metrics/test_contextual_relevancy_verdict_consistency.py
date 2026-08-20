"""Regression tests for ContextualRelevancyMetric verdict classification.

`_calculate_score` and the reason-generation helpers (`_generate_reason` /
`_a_generate_reason`) used to classify verdicts by comparing the raw
`verdict` string with two different predicates: the score path only treated
an exact `"yes"` as relevant, while the reason path only treated an exact
`"no"` as irrelevant (everything else fell into the "relevant" `else`
branch). A non-canonical judge response such as `"yes."`, `"yes "`, or
`"yes<stray text>"` was therefore neither an exact `"yes"` nor an exact
`"no"`, so it was scored as irrelevant while the reason text described it as
relevant, producing contradictory output.

Per maintainer feedback on the PR, the fix now goes one step further:
`ContextualRelevancyVerdict.verdict` (and the equivalent field on
`TurnContextualRelevancyMetric`) is a `bool`, and both the score and the
reason-generation logic branch directly on that boolean. This removes the
string-parsing ambiguity entirely, since `verdict.verdict` can only ever be
`True` or `False`. These tests drive the metric through
`measure()`/`a_measure()` with a stub judge (no network, no API key) and
assert that the score and the statements handed to the reason prompt always
agree.
"""

import asyncio

import pytest

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy.schema import (
    ContextualRelevancyScoreReason,
    ContextualRelevancyVerdicts,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class _StubJudge(DeepEvalBaseLLM):
    """Always returns the same boolean verdict for every statement, and
    records whether the reason-generation prompt treated the statements
    as relevant or irrelevant."""

    def __init__(self, verdict_value: bool):
        self.verdict_value = verdict_value
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
                        "verdict": self.verdict_value,
                        "reason": None,
                    }
                    for i in range(2)
                ]
            )
        self.captured_reason_prompts.append(str(prompt))
        return ContextualRelevancyScoreReason(reason="stub reason")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)


def test_true_verdicts_score_and_reason_agree_sync():
    judge = _StubJudge(True)
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


def test_true_verdicts_score_and_reason_agree_async():
    judge = _StubJudge(True)
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


def test_false_verdict_is_irrelevant():
    judge = _StubJudge(False)
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


def test_verdict_schema_rejects_non_boolean_strings():
    from pydantic import ValidationError

    from deepeval.metrics.contextual_relevancy.schema import (
        ContextualRelevancyVerdict,
    )

    with pytest.raises(ValidationError):
        ContextualRelevancyVerdict(statement="s", verdict="yes.")
    with pytest.raises(ValidationError):
        ContextualRelevancyVerdict(statement="s", verdict="yesshouldfail")

    assert (
        ContextualRelevancyVerdict(statement="s", verdict=True).verdict is True
    )
    assert (
        ContextualRelevancyVerdict(statement="s", verdict=False).verdict
        is False
    )
