import pytest

from deepeval.metrics.conversation_completeness.conversation_completeness import (
    ConversationCompletenessMetric,
)
from deepeval.metrics.conversation_completeness.schema import (
    ConversationCompletenessVerdict,
    ConversationCompletenessScoreReason,
)
from tests.test_core.stubs import DummyWithReasonModel


def test_conversation_completeness_calculate_score_handles_none_verdicts():
    """
    Regression test for GitHub issue #2327.

    If self.verdicts contains None (e.g., due to a failed LLM parse),
    _calculate_score() should not crash with AttributeError on verdict.verdict.
    """

    # Bypass __init__ so we don't depend on initialize_model.
    metric = ConversationCompletenessMetric.__new__(
        ConversationCompletenessMetric
    )

    metric.strict_mode = False
    metric.threshold = 0.5

    # One None verdict (buggy case) and one valid verdict.
    metric.verdicts = [
        None,
        ConversationCompletenessVerdict(verdict="yes", reason="ok"),
    ]

    # On current main this will raise AttributeError: 'NoneType' has no attribute 'verdict'.
    score = metric._calculate_score()

    # After the fix, we expect a sane score and no crash.
    assert isinstance(score, float)
    # With 2 verdict slots, one "yes" and one None treated as non-"no",
    # or as unsatisfied, you can decide semantics – here we assume
    # None is treated as unsatisfied, so the score is 0.5.
    assert score == pytest.approx(0.5)


def test_conversation_completeness_generate_reason_ignores_none_verdicts():
    """
    If self.verdicts contains None, _generate_reason() should skip it
    instead of crashing when accessing verdict.verdict / verdict.reason.
    """

    metric = ConversationCompletenessMetric.__new__(
        ConversationCompletenessMetric
    )

    metric.include_reason = True
    metric.using_native_model = True
    metric.model = DummyWithReasonModel(
        reason_text="conversation completeness reason",
        schema_cls=ConversationCompletenessScoreReason,
    )
    metric.evaluation_cost = 0.0

    metric.score = 0.5
    metric.user_intentions = ["user wants help with X"]

    # One None verdict and one explicit 'no' verdict with a reason.
    metric.verdicts = [
        None,
        ConversationCompletenessVerdict(
            verdict="no", reason="LLM never addressed the user request."
        ),
    ]

    # On current main this will crash on the None entry.
    reason = metric._generate_reason()

    assert isinstance(reason, str)
    assert (
        metric.model.calls
    ), "Expected model.generate to be called for the reason."


@pytest.mark.asyncio
async def test_conversation_completeness_async_generate_reason_ignores_none_verdicts():
    metric = ConversationCompletenessMetric.__new__(
        ConversationCompletenessMetric
    )

    metric.include_reason = True
    metric.using_native_model = True
    metric.model = DummyWithReasonModel(
        reason_text="conversation completeness reason",
        schema_cls=ConversationCompletenessScoreReason,
    )
    metric.evaluation_cost = 0.0

    metric.score = 0.5
    metric.user_intentions = ["user wants help with X"]

    metric.verdicts = [
        None,
        ConversationCompletenessVerdict(
            verdict="no", reason="LLM never addressed the user request."
        ),
    ]

    # On current main this will crash on the None entry.
    reason = await metric._a_generate_reason()

    assert isinstance(reason, str)
    assert metric.model.calls
