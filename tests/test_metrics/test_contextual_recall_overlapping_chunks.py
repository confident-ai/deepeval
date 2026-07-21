import pytest
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import RetrievedContextData


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