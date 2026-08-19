"""Regression tests for ContextualRelevancyMetric verdict parity (#3079).

The metric classifies each verdict twice -- ``_calculate_score`` asks
``verdict.verdict.lower() == "yes"`` while ``_generate_reason`` /
``_a_generate_reason`` invert that and ask ``== "no"``. The first branch
treats anything that is *not* the literal string ``"yes"`` as irrelevant; the
second branch treats anything that is *not* the literal string ``"no"`` as
relevant. Any non-canonical verdict (e.g. ``"yes."`` from a noisy judge) lands
on opposite sides of the two branches and the metric contradicts itself.

These tests drive ``measure`` end-to-end through a stubbed judge that emits
just the verdict token under test, then assert that score and reason agree
on the *bottom line* -- the statements are classified the same way by both
methods. They run without any external API key, so they belong in this
file rather than the existing integration test which guards on
``OPENAI_API_KEY``.
"""

import pytest

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy.contextual_relevancy import (
    _verdict_is_positive,
    _verdict_is_negative,
)
from deepeval.metrics.contextual_relevancy.schema import (
    ContextualRelevancyScoreReason,
    ContextualRelevancyVerdicts,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


NUM_STATEMENTS = 4


class StubJudge(DeepEvalBaseLLM):
    """Stubbed judge that emits a fixed verdict token for every statement.

    Mirrors the off-line reproduction in #3079. Returns a
    ``ContextualRelevancyVerdicts`` shape for the verdict call and a stub
    ``ContextualRelevancyScoreReason`` for the reason call -- no schema is
    parsed, no network is hit.
    """

    def __init__(self, verdict_token: str):
        self.verdict_token = verdict_token

    def load_model(self):
        return self

    def get_model_name(self):
        return "stub-court-of-disagreement"

    def generate(self, prompt, schema=None, **kwargs):
        if schema is ContextualRelevancyVerdicts:
            return ContextualRelevancyVerdicts(
                verdicts=[
                    {
                        "statement": f"statement {i}",
                        "verdict": self.verdict_token,
                        "reason": None,
                    }
                    for i in range(NUM_STATEMENTS)
                ]
            )
        return ContextualRelevancyScoreReason(reason="stub reason")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)





class TestHelpers:
    """Direct unit-tests for the parity helpers themselves."""

    @pytest.mark.parametrize(
        "token",
        ["yes", "YES", "Yes", "yes.", "yes ", " yes", "yesշո", "yes‹junk"],
    )
    def test_positive_helper_accepts_noisy_tokens(self, token: str):
        assert _verdict_is_positive(token) is True

    @pytest.mark.parametrize("token", ["no", "NO", "No", "no.", "no ", " no"])
    def test_negative_helper_accepts_noisy_tokens(self, token: str):
        assert _verdict_is_negative(token) is True

    @pytest.mark.parametrize("token", ["", "maybe", "?yes", "y", "n", "true", "false"])
    def test_helpers_disagree_on_ambiguous_tokens(self, token: str):
        # Ambiguous tokens must NOT be classified as positive OR negative;
        # otherwise one branch will claim them and the metric still
        # contradicts itself.
        assert _verdict_is_positive(token) is False
        assert _verdict_is_negative(token) is False


class TestVerdictParity:
    """End-to-end parity: score and reason agree on every noisy verdict."""

    @pytest.mark.parametrize(
        "token",
        ["yes", "yes ", "yes.", "yesշո"],
    )
    def test_sync_score_one_for_noisy_yes_tokens(self, token: str):
        # Before the fix, only the first row reported ``score == 1.00``;
        # the other three reported ``score == 0.00`` even though the
        # reason branch filed their statements as relevant (#3079).
        metric = ContextualRelevancyMetric(
            model=StubJudge(token), async_mode=False, include_reason=True
        )
        metric.measure(
            LLMTestCase(
                input="what is x?",
                actual_output="x is y",
                retrieval_context=["some context"],
            ),
            _show_indicator=False,
        )
        assert metric.score == pytest.approx(1.0), (
            f"verdict token {token!r} should classify all "
            f"{NUM_STATEMENTS} statements as relevant"
        )

    @pytest.mark.parametrize(
        "token",
        ["yes", "yes ", "yes.", "yesշո"],
    )
    def test_async_score_one_for_noisy_yes_tokens(self, token: str):
        metric = ContextualRelevancyMetric(
            model=StubJudge(token), async_mode=True, include_reason=True
        )
        metric.measure(
            LLMTestCase(
                input="what is x?",
                actual_output="x is y",
                retrieval_context=["some context"],
            ),
            _show_indicator=False,
        )
        assert metric.score == pytest.approx(1.0)

    @pytest.mark.parametrize("token", ["no", "no.", "no ", "NO"])
    def test_sync_score_zero_for_noisy_no_tokens(self, token: str):
        metric = ContextualRelevancyMetric(
            model=StubJudge(token), async_mode=False, include_reason=True
        )
        metric.measure(
            LLMTestCase(
                input="what is x?",
                actual_output="x is y",
                retrieval_context=["some context"],
            ),
            _show_indicator=False,
        )
        assert metric.score == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "token",
        ["yes", "yes ", "yes.", "yesշո", "yes ‹non-Latin fragment›"],
    )
    def test_score_and_reason_branches_agree_on_noisy_yes(self, token: str):
        """The headline behaviour for #3079: score and reason agree.

        Drive the metric end-to-end with a stub judge emitting the given
        verdict token for every statement, then re-run each branch's
        inner loop on the same ``verdicts_list`` to confirm both branches
        classify every statement identically. Before the fix the two
        branches used *different* predicates (``== "yes"`` vs
        ``== "no"``) so the same ``"yes."`` token was counted as not
        relevant by the score branch *and* filed as relevant by the
        reason branch -- the metric contradicted itself.
        """
        metric = ContextualRelevancyMetric(
            model=StubJudge(token), include_reason=True, async_mode=False
        )
        metric.measure(
            LLMTestCase(
                input="what is x?",
                actual_output="x is y",
                retrieval_context=["some context"],
            ),
            _show_indicator=False,
        )
        # Score-branch view: count of positive verdicts == metric.score * NUM_STATEMENTS.
        score_relevant = sum(
            1
            for verdicts in metric.verdicts_list
            for v in verdicts.verdicts
            if _verdict_is_positive(v.verdict)
        )
        # Reason-branch view: every non-negative verdict filed as RELEVANT.
        reason_relevant = sum(
            1
            for verdicts in metric.verdicts_list
            for v in verdicts.verdicts
            if not _verdict_is_negative(v.verdict)
        )
        assert score_relevant == reason_relevant == NUM_STATEMENTS, (
            f"verdict {token!r}: branches disagree -- "
            f"score says {score_relevant}/{NUM_STATEMENTS} relevant, "
            f"reason says {reason_relevant}/{NUM_STATEMENTS} relevant"
        )
        assert metric.score == pytest.approx(1.0)
        assert metric.reason == "stub reason"

    @pytest.mark.parametrize("token", ["no", "no.", "no ", "NO"])
    def test_score_and_reason_branches_agree_on_noisy_no(self, token: str):
        """Mirror of the noisy-'yes' case for the negative branch.

        Mirrors the inner loops of ``_calculate_score`` and
        ``_generate_reason`` post-measure. For every ambiguous ``no`` token
        the two branches must count identically (== 0) since the helpers
        agree on whether the token is ``no`` up to its noisy suffix.
        """
        metric = ContextualRelevancyMetric(
            model=StubJudge(token), include_reason=True, async_mode=False
        )
        metric.measure(
            LLMTestCase(
                input="what is x?",
                actual_output="x is y",
                retrieval_context=["some context"],
            ),
            _show_indicator=False,
        )
        score_relevant = sum(
            1
            for verdicts in metric.verdicts_list
            for v in verdicts.verdicts
            if _verdict_is_positive(v.verdict)
        )
        reason_relevant = sum(
            1
            for verdicts in metric.verdicts_list
            for v in verdicts.verdicts
            if not _verdict_is_negative(v.verdict)
        )
        assert score_relevant == reason_relevant == 0
        assert metric.score == pytest.approx(0.0)
