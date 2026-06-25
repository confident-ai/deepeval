"""Tests for ContextualRecallMetric with overlapping-chunk retrieval scenarios.

These fixtures document the failure mode described in issue #2788:
ContextualRecallMetric over-penalises overlapping chunks because it lacks the
source-grouping step added to ContextualPrecisionMetric in PR #2743.

The tests are written as regression targets for the proposed fix: applying
RetrievedContextData.source-based grouping symmetrically to the recall metric.

Requires OPENAI_API_KEY (or any LLM configured in conftest.py).
"""

import pytest
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def overlapping_revenue_chunks():
    """Two same-source overlapping chunks covering the same revenue statement.

    Standard 20% overlap from sliding-window chunking of a 10-K MD&A section.
    Both chunks support the expected output; only one should be scored.
    """
    chunk_a = (
        "Revenue for FY2023 was $4.2 billion, up 12% year-over-year, "
        "driven by strong performance in the cloud segment."
    )
    chunk_b = (
        "Driven by strong performance in the cloud segment, revenue for "
        "FY2023 reached $4.2 billion representing a 12% increase year-over-year."
    )
    return [
        RetrievedContextData(
            content=chunk_a,
            source="10k_2023.pdf/MD&A/p14",
        ),
        RetrievedContextData(
            content=chunk_b,
            source="10k_2023.pdf/MD&A/p14",
        ),
    ]


@pytest.fixture
def multi_statement_expected_output():
    """Expected output with three independent factual statements.

    Validates that recall scoring works correctly when:
    - Statement 1 is covered by chunk from source A
    - Statement 2 is covered by overlapping chunks from source A
    - Statement 3 is covered by chunk from source B
    """
    return (
        "Revenue for FY2023 was $4.2 billion, up 12% year-over-year. "
        "The growth was driven by cloud segment performance. "
        "Net income was $320 million."
    )


@pytest.fixture
def retrieval_context_multi_source():
    """Three chunks: two overlapping from 10-K, one from earnings call."""
    return [
        RetrievedContextData(
            content=(
                "Revenue for FY2023 was $4.2 billion, up 12% year-over-year, "
                "driven by strong cloud segment performance."
            ),
            source="10k_2023.pdf/MD&A/p14",
        ),
        RetrievedContextData(
            content=(
                "Strong cloud segment performance drove revenue to $4.2 billion "
                "for FY2023, a 12% year-over-year increase."
            ),
            source="10k_2023.pdf/MD&A/p14",
        ),
        RetrievedContextData(
            content="Net income for FY2023 was $320 million.",
            source="earnings_call_q4.pdf/transcript/p3",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestContextualRecallOverlappingChunks:
    """Regression suite for overlapping-chunk scoring in ContextualRecallMetric.

    These tests document the failure mode from issue #2788. Once the symmetric
    source-grouping fix is applied (parallel to PR #2743 for ContextualPrecision),
    all tests in this class should pass.
    """

    def test_same_source_overlap_does_not_lower_recall(
        self, overlapping_revenue_chunks
    ):
        """Recall should remain high when overlapping chunks both cover the
        expected statement. The metric should not penalise redundancy.

        Failure mode (pre-fix): LLM returns 'yes' for chunk_a and 'no' for
        chunk_b (partial redundancy), halving the recall score even though
        the expected output is fully covered.

        Expected behaviour (post-fix): same-source chunks are grouped;
        merged content is scored once → recall ~1.0.
        """
        test_case = LLMTestCase(
            input="What was revenue for FY2023 and what drove it?",
            actual_output=(
                "Revenue for FY2023 was $4.2 billion, up 12% year-over-year, "
                "driven by strong cloud segment performance."
            ),
            expected_output=(
                "Revenue for FY2023 was $4.2 billion, up 12% year-over-year."
            ),
            retrieval_context=[
                c.content for c in overlapping_revenue_chunks
            ],
            additional_metadata={
                "retrieved_context_data": overlapping_revenue_chunks
            },
        )
        metric = ContextualRecallMetric(threshold=0.8, verbose_mode=False)
        metric.measure(test_case)
        assert metric.score >= 0.8, (
            f"Expected recall >= 0.8 for fully-covered expected output with "
            f"same-source overlapping chunks, got {metric.score:.3f}. "
            f"Source grouping may not be applied to ContextualRecallMetric "
            f"(see issue #2788)."
        )

    def test_multi_source_recall_not_inflated_by_overlap(
        self, multi_statement_expected_output, retrieval_context_multi_source
    ):
        """When multiple expected statements are present, recall should
        correctly attribute coverage without being distorted by same-source
        overlapping chunks.

        All three expected statements are covered in the retrieval context.
        Recall should be high (~1.0), not artificially deflated by the
        overlapping 10-K chunks being scored independently.
        """
        test_case = LLMTestCase(
            input="Summarise FY2023 financial results.",
            actual_output=(
                "FY2023 revenue was $4.2B (+12% YoY) driven by cloud. "
                "Net income was $320M."
            ),
            expected_output=multi_statement_expected_output,
            retrieval_context=[
                c.content for c in retrieval_context_multi_source
            ],
            additional_metadata={
                "retrieved_context_data": retrieval_context_multi_source
            },
        )
        metric = ContextualRecallMetric(threshold=0.8, verbose_mode=False)
        metric.measure(test_case)
        assert metric.score >= 0.8, (
            f"Expected recall >= 0.8 when all expected statements are covered "
            f"across retrieval chunks, got {metric.score:.3f}."
        )

    def test_increasing_overlap_does_not_decrease_recall(self):
        """Adding a second overlapping chunk (same source) should not lower recall.

        Validates the core invariant: more context from the same source should
        never make recall worse. This is the monotonicity property that source
        grouping preserves.

        Pre-fix: second overlapping chunk can receive 'no' verdict → lower score.
        Post-fix: chunks are grouped → single 'yes' verdict → stable score.
        """
        base_chunk = (
            "The company reported net income of $320 million for FY2023."
        )
        overlap_chunk = (
            "For FY2023, the company's net income was $320 million, "
            "reflecting continued operational efficiency."
        )
        expected = "Net income for FY2023 was $320 million."

        def make_case(use_overlap: bool):
            context = [base_chunk]
            rcd = [
                RetrievedContextData(
                    content=base_chunk,
                    source="10k_2023.pdf/FinancialStatements/p42",
                )
            ]
            if use_overlap:
                context.append(overlap_chunk)
                rcd.append(
                    RetrievedContextData(
                        content=overlap_chunk,
                        source="10k_2023.pdf/FinancialStatements/p42",
                    )
                )
            return LLMTestCase(
                input="What was net income for FY2023?",
                actual_output="Net income for FY2023 was $320 million.",
                expected_output=expected,
                retrieval_context=context,
                additional_metadata={"retrieved_context_data": rcd},
            )

        metric = ContextualRecallMetric(threshold=0.0, verbose_mode=False)

        metric.measure(make_case(use_overlap=False))
        score_without_overlap = metric.score

        metric.measure(make_case(use_overlap=True))
        score_with_overlap = metric.score

        assert score_with_overlap >= score_without_overlap - 0.05, (
            f"Adding a same-source overlapping chunk should not significantly "
            f"decrease recall. Without overlap: {score_without_overlap:.3f}, "
            f"with overlap: {score_with_overlap:.3f}. "
            f"Delta exceeds 5% tolerance (see issue #2788)."
        )
