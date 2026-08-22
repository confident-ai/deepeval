"""Regression tests for TurnContextualRelevancyMetric verdict classification.

Companion to `test_contextual_relevancy_verdict_consistency.py`. Following
maintainer feedback on PR #3080, `ContextualRelevancyVerdict.verdict` (the
model used by both `ContextualRelevancyMetric` and
`TurnContextualRelevancyMetric`) was changed from a `"yes"`/`"no"` string to
a `bool`, and the metric's scoring/reason logic now branches directly on
that boolean instead of string-comparing it. These tests drive
`TurnContextualRelevancyMetric` through `measure()`/`a_measure()` with a
stub judge (no network, no API key) and assert that the score and the
statements handed to the reason prompt agree with the boolean verdicts
returned by the judge.
"""

import asyncio

import pytest

from deepeval.metrics import TurnContextualRelevancyMetric
from deepeval.metrics.turn_contextual_relevancy.schema import (
    ContextualRelevancyScoreReason,
    ContextualRelevancyVerdicts,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


class _StubJudge(DeepEvalBaseLLM):
    """Always returns the same boolean verdict for every statement, and
    records the interaction-level reason prompts it was asked to fill in."""

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


def _test_case():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="what is x?"),
            Turn(
                role="assistant",
                content="x is y",
                retrieval_context=["some context"],
            ),
        ]
    )


def test_true_verdicts_score_and_reason_agree_sync():
    judge = _StubJudge(True)
    metric = TurnContextualRelevancyMetric(
        model=judge, async_mode=False, include_reason=True
    )

    metric.measure(_test_case(), _show_indicator=False)

    assert metric.score == 1.0
    # first captured prompt is the interaction-level reason, statements
    # judged True should appear as relevant statements
    reason_prompt = judge.captured_reason_prompts[0]
    assert "statement 0" in reason_prompt
    assert "statement 1" in reason_prompt


def test_true_verdicts_score_and_reason_agree_async():
    judge = _StubJudge(True)
    metric = TurnContextualRelevancyMetric(
        model=judge, async_mode=True, include_reason=True
    )

    asyncio.run(metric.a_measure(_test_case(), _show_indicator=False))

    assert metric.score == 1.0
    reason_prompt = judge.captured_reason_prompts[0]
    assert "statement 0" in reason_prompt
    assert "statement 1" in reason_prompt


def test_false_verdict_is_irrelevant():
    judge = _StubJudge(False)
    metric = TurnContextualRelevancyMetric(
        model=judge, async_mode=False, include_reason=True
    )

    metric.measure(_test_case(), _show_indicator=False)

    assert metric.score == 0.0
    reason_prompt = judge.captured_reason_prompts[0]
    # irrelevant statements are reported as "<statement>: <reason>" in a
    # separate section from the relevant-statements list
    assert (
        "Statement in the retrieval context that is relevant" in reason_prompt
    )
    relevant_section = reason_prompt.split(
        "Statement in the retrieval context that is relevant"
    )[1]
    assert "statement 0" not in relevant_section
    assert "statement 1" not in relevant_section


def test_verdict_schema_rejects_non_boolean_strings():
    from pydantic import ValidationError

    from deepeval.metrics.turn_contextual_relevancy.schema import (
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
