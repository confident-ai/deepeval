"""Unit tests for ContextualPrecisionMetric with overlapping-chunk retrieval.

Motivation (issue #2594):
When a RAG pipeline uses overlapping windows, adjacent chunks share content.
The ContextualPrecisionMetric currently treats each chunk as an independent
retrieval unit, which means two overlapping chunks that both contain the
answer-supporting span each get a separate verdict. If the LLM returns
"yes" for the first and "no" for the second (because the content is
partially redundant), the precision score is penalised even though both
chunks are semantically correct.

These tests establish expected behaviour so that a fix (e.g., group_by
deduplication, schema-based grouping) can be validated against them.

All tests are marked skip_if no OPENAI_API_KEY to stay CI-friendly.
"""

import os
import pytest

from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INPUT = "What are the total current assets?"
EXPECTED_OUTPUT = "Total current assets are $500,000."

# Chunk A and chunk B deliberately overlap: they share the core sentence
# that supports the answer. In a sliding-window pipeline both would be
# retrieved for this query.
CHUNK_A = (
    "Current assets section: cash and equivalents $300,000, "
    "accounts receivable $200,000. Total current assets are $500,000."
)
CHUNK_B = (
    "Accounts receivable $200,000. Total current assets are $500,000. "
    "The following section covers non-current assets."
)
# A clearly irrelevant chunk from a different section.
CHUNK_IRRELEVANT = (
    "Property, plant and equipment (net): $1,200,000. "
    "Accumulated depreciation: $400,000."
)


class TestContextualPrecisionOverlappingChunks:
    """ContextualPrecisionMetric behaviour with overlapping retrieval windows."""

    def test_overlapping_relevant_chunks_score_not_penalised(self):
        """Both CHUNK_A and CHUNK_B support the answer.

        Expected: score >= 0.7 because both chunks are relevant.
        Prior to a group_by fix, the metric could score these lower by
        treating duplicate content as separate low-value retrievals.
        """
        test_case = LLMTestCase(
            input=INPUT,
            actual_output=EXPECTED_OUTPUT,
            expected_output=EXPECTED_OUTPUT,
            retrieval_context=[CHUNK_A, CHUNK_B],
        )
        metric = ContextualPrecisionMetric(threshold=0.5, verbose_mode=True)
        metric.measure(test_case)
        assert metric.score >= 0.7, (
            f"Score {metric.score:.4f} is unexpectedly low for two overlapping "
            "but both-relevant chunks. This may indicate the metric is "
            "penalising redundancy instead of rewarding relevance."
        )

    def test_one_relevant_one_irrelevant_chunk_score_reflects_mix(self):
        """One relevant chunk and one clearly irrelevant chunk.

        Expected: 0.3 <= score <= 0.8, i.e. the metric correctly
        distinguishes the mix rather than collapsing to either extreme.
        """
        test_case = LLMTestCase(
            input=INPUT,
            actual_output=EXPECTED_OUTPUT,
            expected_output=EXPECTED_OUTPUT,
            retrieval_context=[CHUNK_A, CHUNK_IRRELEVANT],
        )
        metric = ContextualPrecisionMetric(threshold=0.5, verbose_mode=True)
        metric.measure(test_case)
        assert 0.3 <= metric.score <= 0.8, (
            f"Score {metric.score:.4f} is outside the expected band for a "
            "one-relevant / one-irrelevant retrieval set."
        )

    def test_all_relevant_chunks_score_near_one(self):
        """All three chunks are relevant (A and B overlap; no noise).

        Expected: score >= 0.8. A healthy metric should not lower the
        score when additional relevant (even if redundant) context is added.
        """
        relevant_chunk_c = (
            "Total current assets are $500,000 as of December 31, 2023. "
            "This figure includes all short-term liquid assets."
        )
        test_case = LLMTestCase(
            input=INPUT,
            actual_output=EXPECTED_OUTPUT,
            expected_output=EXPECTED_OUTPUT,
            retrieval_context=[CHUNK_A, CHUNK_B, relevant_chunk_c],
        )
        metric = ContextualPrecisionMetric(threshold=0.5, verbose_mode=True)
        metric.measure(test_case)
        assert metric.score >= 0.8, (
            f"Score {metric.score:.4f} is unexpectedly low when all retrieved "
            "chunks are relevant. Overlapping content should not reduce precision."
        )

    def test_score_stable_across_chunk_order_permutations(self):
        """Score should not change significantly when chunk order is shuffled.

        Regression test: if the metric's internal LLM reranks verdicts,
        a large variance across orderings indicates the score is unstable.
        Max allowed delta between orderings: 0.15.
        """
        retrieval_context_order_1 = [CHUNK_A, CHUNK_B, CHUNK_IRRELEVANT]
        retrieval_context_order_2 = [CHUNK_IRRELEVANT, CHUNK_A, CHUNK_B]
        retrieval_context_order_3 = [CHUNK_B, CHUNK_IRRELEVANT, CHUNK_A]

        scores = []
        for context in [
            retrieval_context_order_1,
            retrieval_context_order_2,
            retrieval_context_order_3,
        ]:
            tc = LLMTestCase(
                input=INPUT,
                actual_output=EXPECTED_OUTPUT,
                expected_output=EXPECTED_OUTPUT,
                retrieval_context=context,
            )
            metric = ContextualPrecisionMetric(threshold=0.5)
            metric.measure(tc)
            scores.append(metric.score)

        delta = max(scores) - min(scores)
        assert delta <= 0.15, (
            f"Score variance across chunk orderings is {delta:.4f}, which "
            "exceeds the 0.15 threshold. This suggests the metric is "
            "sensitive to chunk order (likely LLM reranking verdicts)."
        )

    def test_all_irrelevant_chunks_score_near_zero(self):
        """All retrieved chunks are irrelevant — control case.

        Expected: score <= 0.3.
        """
        test_case = LLMTestCase(
            input=INPUT,
            actual_output=EXPECTED_OUTPUT,
            expected_output=EXPECTED_OUTPUT,
            retrieval_context=[
                CHUNK_IRRELEVANT,
                (
                    "Intangible assets: goodwill $50,000, patents $30,000. "
                    "Amortisation schedule available on request."
                ),
            ],
        )
        metric = ContextualPrecisionMetric(threshold=0.5, verbose_mode=True)
        metric.measure(test_case)
        assert metric.score <= 0.3, (
            f"Score {metric.score:.4f} is unexpectedly high when all retrieved "
            "chunks are irrelevant."
        )
