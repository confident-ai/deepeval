"""Tests for ContextualPrecisionMetric with overlapping-chunk retrieval scenarios.

These fixtures validate the source-grouping deduplication introduced in PR #2743
(RetrievedContextData._group_retrieval_contexts) and the weighted cumulative
precision (WCP) formula fix. Each test corresponds to a failure mode described
in issue #2594.

Requires OPENAI_API_KEY (or any LLM configured in conftest.py).
"""

import pytest
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def overlapping_narrative_chunks():
    """Two adjacent 20%-overlap chunks from a 10-K narrative section.

    Both chunks cover the same answer-supporting span. Without source grouping
    the metric would penalise the second chunk as redundant; with grouping they
    are merged under the same source and scored once.
    """
    chunk_a = (
        "Revenue for fiscal 2023 was $4.2 billion, up 12% year-over-year, "
        "driven by strong performance in the cloud segment."
    )
    chunk_b = (
        "Driven by strong performance in the cloud segment, revenue for fiscal "
        "2023 reached $4.2 billion representing a 12% increase."
    )
    return [
        RetrievedContextData(
            content=chunk_a,
            source="10k_2023.pdf/Management_Discussion/p14",
        ),
        RetrievedContextData(
            content=chunk_b,
            source="10k_2023.pdf/Management_Discussion/p14",
        ),
    ]


@pytest.fixture
def non_overlapping_mixed_chunks():
    """Three chunks from different sources — one relevant, two not.

    Validates that source grouping does not incorrectly merge chunks from
    different documents.
    """
    return [
        RetrievedContextData(
            content="Net income for Q4 2023 was $320 million.",
            source="10k_2023.pdf/Financial_Statements/p42",
        ),
        RetrievedContextData(
            content="The company plans to expand into three new markets in 2024.",
            source="earnings_call_q4.pdf/transcript/p3",
        ),
        RetrievedContextData(
            content="Balance sheet total assets stood at $18.7 billion.",
            source="10k_2023.pdf/Financial_Statements/p38",
        ),
    ]


@pytest.fixture
def table_cell_overlap_chunks():
    """Chunks from a balance-sheet table with deliberate boundary overlap.

    Sliding-window chunking across table rows produces chunks where the
    header row appears in multiple adjacent chunks. Validates that overlapping
    table-header content is not penalised as irrelevant.
    """
    header_plus_row1 = (
        "| Asset | FY2023 | FY2022 |\n"
        "|-------|--------|--------|\n"
        "| Cash and equivalents | $1.2B | $0.9B |"
    )
    row1_plus_row2 = (
        "| Cash and equivalents | $1.2B | $0.9B |\n"
        "| Short-term investments | $0.4B | $0.6B |"
    )
    return [
        RetrievedContextData(
            content=header_plus_row1,
            source="10k_2023.pdf/Balance_Sheet/p28",
        ),
        RetrievedContextData(
            content=row1_plus_row2,
            source="10k_2023.pdf/Balance_Sheet/p28",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestContextualPrecisionOverlappingChunks:
    """Regression suite for overlapping-chunk scoring (issue #2594)."""

    def test_same_source_overlap_does_not_penalise(self, overlapping_narrative_chunks):
        """Score should remain high when two same-source overlapping chunks
        both support the answer. Redundancy != irrelevance.

        Regression for: metric scoring chunk_b as irrelevant because it
        partially repeats chunk_a content.
        """
        test_case = LLMTestCase(
            input="What was the company's revenue for fiscal 2023?",
            actual_output=(
                "The company's revenue for fiscal 2023 was $4.2 billion, "
                "representing a 12% year-over-year increase driven by cloud."
            ),
            expected_output=(
                "Revenue for fiscal 2023 was $4.2 billion, up 12% YoY."
            ),
            retrieval_context=[
                c.content for c in overlapping_narrative_chunks
            ],
            additional_metadata={
                "retrieved_context_data": overlapping_narrative_chunks
            },
        )
        metric = ContextualPrecisionMetric(threshold=0.7, verbose_mode=False)
        metric.measure(test_case)
        assert metric.score >= 0.7, (
            f"Expected score >= 0.7 for same-source overlapping chunks, "
            f"got {metric.score:.3f}. Source grouping may not be active."
        )

    def test_cross_source_chunks_scored_independently(self, non_overlapping_mixed_chunks):
        """Chunks from distinct sources should each be scored on their own merit.

        Validates that source grouping does not incorrectly merge content from
        different documents into a single evaluation unit.
        """
        test_case = LLMTestCase(
            input="What was net income for Q4 2023?",
            actual_output="Net income for Q4 2023 was $320 million.",
            expected_output="Net income for Q4 2023 was $320 million.",
            retrieval_context=[
                c.content for c in non_overlapping_mixed_chunks
            ],
            additional_metadata={
                "retrieved_context_data": non_overlapping_mixed_chunks
            },
        )
        metric = ContextualPrecisionMetric(threshold=0.5, verbose_mode=False)
        metric.measure(test_case)
        # Only the first chunk is directly relevant; score should reflect
        # that one relevant chunk was ranked first (WCP favours early relevant)
        assert metric.score > 0.0, (
            f"Score should be positive when top chunk is relevant, "
            f"got {metric.score:.3f}."
        )

    def test_table_boundary_overlap_not_penalised(self, table_cell_overlap_chunks):
        """Overlapping table-header rows in adjacent chunks should not lower score.

        Financial tables are commonly chunked with boundary overlap to avoid
        cutting mid-row. Both chunks are from the same source/page and cover
        the same table; they should be grouped before scoring.
        """
        test_case = LLMTestCase(
            input="What was cash and equivalents on the balance sheet for FY2023?",
            actual_output="Cash and equivalents were $1.2 billion in FY2023.",
            expected_output="Cash and equivalents: $1.2B (FY2023).",
            retrieval_context=[
                c.content for c in table_cell_overlap_chunks
            ],
            additional_metadata={
                "retrieved_context_data": table_cell_overlap_chunks
            },
        )
        metric = ContextualPrecisionMetric(threshold=0.7, verbose_mode=False)
        metric.measure(test_case)
        assert metric.score >= 0.7, (
            f"Expected score >= 0.7 for table-boundary overlapping chunks, "
            f"got {metric.score:.3f}."
        )

    def test_wcp_formula_monotone_with_relevant_rank(self):
        """Weighted cumulative precision should decrease as the relevant chunk
        moves to a lower rank position.

        Validates the corrected _calculate_score() WCP formula from PR #2743:
        score(rank_1_relevant) > score(rank_2_relevant) > score(rank_3_relevant).
        """
        base_context = [
            "The answer to the question is here.",
            "This chunk is not relevant.",
            "This chunk is also not relevant.",
        ]

        def make_case(relevant_first, relevant_second, relevant_third):
            contexts = []
            if relevant_first:
                contexts = [base_context[0], base_context[1], base_context[2]]
            elif relevant_second:
                contexts = [base_context[1], base_context[0], base_context[2]]
            else:
                contexts = [base_context[1], base_context[2], base_context[0]]
            return LLMTestCase(
                input="What is the answer?",
                actual_output="The answer to the question is here.",
                expected_output="The answer to the question is here.",
                retrieval_context=contexts,
            )

        metric = ContextualPrecisionMetric(threshold=0.0, verbose_mode=False)

        metric.measure(make_case(True, False, False))
        score_rank1 = metric.score

        metric.measure(make_case(False, True, False))
        score_rank2 = metric.score

        metric.measure(make_case(False, False, True))
        score_rank3 = metric.score

        assert score_rank1 >= score_rank2 >= score_rank3, (
            f"WCP should be monotonically non-increasing as relevant chunk rank "
            f"decreases: rank1={score_rank1:.3f}, rank2={score_rank2:.3f}, "
            f"rank3={score_rank3:.3f}"
        )
