from unittest.mock import patch, MagicMock
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData


class TestContextualRecallOverlappingChunks:
    def test_group_merges_same_source(self):
        metric = ContextualRecallMetric(async_mode=False)
        contexts = [
            RetrievedContextData(
                context="Revenue was 4.2B with 12% growth.",
                source="annual_report.pdf",
            ),
            RetrievedContextData(
                context="Revenue was 4.2B with 12% growth. Net income rose 8%.",
                source="annual_report.pdf",
            ),
        ]
        grouped = metric._group_retrieval_contexts(contexts)
        assert len(grouped) == 1
        assert "annual_report.pdf" in grouped[0]
        assert "4.2B" in grouped[0]

    def test_group_preserves_different_sources(self):
        metric = ContextualRecallMetric(async_mode=False)
        contexts = [
            RetrievedContextData(
                context="Revenue was 4.2B.",
                source="annual_report.pdf",
            ),
            RetrievedContextData(
                context="Expenses were 3.1B.",
                source="expense_report.pdf",
            ),
        ]
        grouped = metric._group_retrieval_contexts(contexts)
        assert len(grouped) == 2

    def test_group_plain_strings_pass_through(self):
        metric = ContextualRecallMetric(async_mode=False)
        contexts = [
            "Revenue was 4.2B with 12% growth.",
            "Revenue was 4.2B with 12% growth.",
        ]
        grouped = metric._group_retrieval_contexts(contexts)
        assert len(grouped) == 2

    def test_group_mixed_types(self):
        metric = ContextualRecallMetric(async_mode=False)
        contexts = [
            RetrievedContextData(
                context="Revenue was 4.2B.",
                source="annual_report.pdf",
            ),
            "Some standalone context string.",
            RetrievedContextData(
                context="Growth was 12%.",
                source="annual_report.pdf",
            ),
        ]
        grouped = metric._group_retrieval_contexts(contexts)
        assert len(grouped) == 2
        assert "annual_report.pdf" in grouped[0]
        assert grouped[1] == "Some standalone context string."

    def test_group_merges_partial_overlap_same_source(self):
        metric = ContextualRecallMetric(async_mode=False)
        contexts = [
            RetrievedContextData(
                context="Revenue for FY2023 was $4.2B, up 12% YoY.",
                source="annual_report.pdf",
            ),
            RetrievedContextData(
                context="Operating expenses increased 8% to $2.1B in FY2023.",
                source="annual_report.pdf",
            ),
        ]
        grouped = metric._group_retrieval_contexts(contexts)
        assert len(grouped) == 1
        assert "4.2B" in grouped[0]
        assert "2.1B" in grouped[0]
        assert "annual_report.pdf" in grouped[0]

    def test_measure_grouped_overlapping_chunks_full_recall(self):
        metric = ContextualRecallMetric(threshold=0.8, async_mode=False)
        test_case = LLMTestCase(
            input="What was revenue for FY2023?",
            actual_output="Revenue was $4.2 billion in FY2023, up 12% year-over-year.",
            expected_output="Revenue for FY2023 was $4.2 billion, up 12% year-over-year.",
            retrieval_context=[
                RetrievedContextData(
                    context="Revenue for FY2023 was $4.2B, up 12% YoY, driven by cloud growth.",
                    source="10k_filing.pdf",
                ),
                RetrievedContextData(
                    context="Driven by cloud growth, revenue for FY2023 reached $4.2B (12% increase).",
                    source="10k_filing.pdf",
                ),
            ],
        )

        grouped = metric._group_retrieval_contexts(test_case.retrieval_context)
        assert len(grouped) == 1, (
            "Two overlapping chunks from the same source must merge into one context"
        )

        mock_verdict = MagicMock()
        mock_verdict.verdict = "yes"
        mock_verdict.expected_output = (
            "Revenue for FY2023 was $4.2 billion, up 12% year-over-year."
        )
        mock_verdict.reason = "The merged context contains both the revenue figure and growth rate."

        mock_generate = MagicMock(return_value=[mock_verdict])

        with (
            patch.object(metric, "_generate_verdicts", mock_generate),
            patch.object(
                metric,
                "_generate_reason",
                return_value="All expected sentences attributed to the merged context.",
            ),
        ):
            score = metric.measure(test_case)

        call_args = mock_generate.call_args
        retrieval_context_passed = call_args[0][1]
        assert len(retrieval_context_passed) == 1, (
            f"Judge must receive 1 merged context, got {len(retrieval_context_passed)}"
        )
        assert "10k_filing.pdf" in retrieval_context_passed[0], (
            "Merged context must contain the source identifier"
        )
        assert score == 1.0, (
            f"With grouped overlapping chunks recall should be 1.0, got {score}"
        )

    def test_measure_ungrouped_overlapping_chunks_penalised(self):
        metric = ContextualRecallMetric(threshold=0.8, async_mode=False)

        verdict_yes = MagicMock()
        verdict_yes.verdict = "yes"

        verdict_no = MagicMock()
        verdict_no.verdict = "no"

        metric.verdicts = [verdict_yes, verdict_no]
        bug_score = metric._calculate_score()

        assert bug_score == 0.5, (
            f"Without grouping, redundant overlap penalises recall to 0.5, got {bug_score}"
        )
        assert bug_score < 1.0, (
            "Bug scenario must produce a penalised score to contrast with the fix"
        )
