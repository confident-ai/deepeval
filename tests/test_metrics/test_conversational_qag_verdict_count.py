"""Regression tests for the conversational QAG score denominators.

Each metric below asks the judge to assess a known number of items
(user intentions, eligible turns, sliding windows, extracted claims),
one verdict per item. A verdict can come back missing: the judge
truncates its list, or a reply lands out of vocabulary and is dropped
to ``None`` during normalization. ``score_qag_verdicts`` used to divide
by the number of verdicts that survived, so every missing assessment
shrank the denominator and inflated the score: 1 assessed item out of
3 scored a perfect 1.0 (issue #3401, follow-up to #3347).

These tests drive the scoring paths directly with a stub model, so
they run without an LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)
from deepeval.metrics.conversation_completeness.schema import (
    ConversationCompletenessVerdict,
)
from deepeval.metrics.knowledge_retention.schema import (
    KnowledgeRetentionVerdict,
)
from deepeval.metrics.turn_faithfulness.schema import FaithfulnessVerdict
from deepeval.metrics.turn_relevancy.schema import TurnRelevancyVerdict
from deepeval.models import DeepEvalBaseLLM


class _StubJudge(DeepEvalBaseLLM):
    """Never called: these tests drive the scoring paths directly."""

    def load_model(self):
        return self

    def get_model_name(self):
        return "stub-judge"

    def generate(self, prompt, schema=None, **kwargs):
        raise AssertionError("stub judge must not be called")

    async def a_generate(self, prompt, schema=None, **kwargs):
        raise AssertionError("stub judge must not be called")


def test_conversation_completeness_dropped_verdicts_count_against_score():
    """3 intentions asked, 1 assessed: score is 1/3, not 1."""
    metric = ConversationCompletenessMetric(
        model=_StubJudge(), include_reason=False
    )
    metric.user_intentions = ["book a flight", "change the date", "refund it"]
    metric.verdicts = [
        ConversationCompletenessVerdict(verdict="yes"),
        None,
        None,
    ]
    assert metric._calculate_score() == pytest.approx(1 / 3)


def test_conversation_completeness_complete_verdicts_unchanged():
    """Control: all 3 intentions assessed, score is unchanged."""
    metric = ConversationCompletenessMetric(
        model=_StubJudge(), include_reason=False
    )
    metric.user_intentions = ["book a flight", "change the date", "refund it"]
    metric.verdicts = [
        ConversationCompletenessVerdict(verdict="yes"),
        ConversationCompletenessVerdict(verdict="yes"),
        ConversationCompletenessVerdict(verdict="no"),
    ]
    assert metric._calculate_score() == pytest.approx(2 / 3)


def test_knowledge_retention_dropped_verdicts_count_against_score():
    """3 eligible turns asked, 1 assessed ("no" passes): score is 1/3."""
    metric = KnowledgeRetentionMetric(model=_StubJudge(), include_reason=False)
    metric.verdicts = [
        KnowledgeRetentionVerdict(verdict="no"),
        None,
        None,
    ]
    assert metric._calculate_score() == pytest.approx(1 / 3)


def test_knowledge_retention_complete_verdicts_unchanged():
    """Control: all 3 turns assessed, 2 retained: score stays 2/3."""
    metric = KnowledgeRetentionMetric(model=_StubJudge(), include_reason=False)
    metric.verdicts = [
        KnowledgeRetentionVerdict(verdict="no"),
        KnowledgeRetentionVerdict(verdict="no"),
        KnowledgeRetentionVerdict(verdict="yes"),
    ]
    assert metric._calculate_score() == pytest.approx(2 / 3)


def test_turn_relevancy_dropped_verdicts_count_against_score():
    """2 windows asked, 1 assessed: score is 1/2, not 1."""
    metric = TurnRelevancyMetric(model=_StubJudge(), include_reason=False)
    metric.verdicts = [TurnRelevancyVerdict(verdict="yes"), None]
    assert metric._calculate_score() == pytest.approx(0.5)


def test_turn_faithfulness_truncated_verdicts_count_against_score():
    """3 claims extracted, judge answered 1: score is 1/3, not 1."""
    metric = TurnFaithfulnessMetric(model=_StubJudge(), include_reason=False)
    score, _ = metric._get_interaction_score_and_reason(
        [FaithfulnessVerdict(verdict="yes")],
        multimodal=False,
        expected_count=3,
    )
    assert score == pytest.approx(1 / 3)


def test_turn_faithfulness_empty_verdicts_with_claims_scores_zero():
    """Claims were extracted but no verdict came back: score is 0."""
    metric = TurnFaithfulnessMetric(model=_StubJudge(), include_reason=False)
    score, _ = metric._get_interaction_score_and_reason(
        [], multimodal=False, expected_count=3
    )
    assert score == 0.0


def test_turn_faithfulness_no_claims_still_scores_one():
    """No claims to verify keeps the existing perfect score."""
    metric = TurnFaithfulnessMetric(model=_StubJudge(), include_reason=False)
    score, _ = metric._get_interaction_score_and_reason(
        [], multimodal=False, expected_count=0
    )
    assert score == 1.0


def test_turn_faithfulness_truncated_verdicts_count_against_score_async():
    """Async path: 3 claims extracted, judge answered 1: score is 1/3."""
    metric = TurnFaithfulnessMetric(model=_StubJudge(), include_reason=False)
    score, _ = asyncio.run(
        metric._a_get_interaction_score_and_reason(
            [FaithfulnessVerdict(verdict="yes")],
            multimodal=False,
            expected_count=3,
        )
    )
    assert score == pytest.approx(1 / 3)


def test_turn_faithfulness_complete_verdicts_unchanged():
    """Control: all 3 claims assessed, 2 supported: score stays 2/3."""
    metric = TurnFaithfulnessMetric(model=_StubJudge(), include_reason=False)
    score, _ = metric._get_interaction_score_and_reason(
        [
            FaithfulnessVerdict(verdict="yes"),
            FaithfulnessVerdict(verdict="yes"),
            FaithfulnessVerdict(verdict="no"),
        ],
        multimodal=False,
        expected_count=3,
    )
    assert score == pytest.approx(2 / 3)
