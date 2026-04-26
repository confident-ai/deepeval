import pytest
from deepeval.metrics import TrustScoreMetric
from deepeval.test_case import LLMTestCase


class TestTrustScoreMetric:

    @pytest.fixture
    def source_tiers(self):
        return {
            "SEC": 1,
            "Bloomberg": 2,
            "NewsSite": 3,
            "Blog": 4,
            "Reddit": 5,
        }

    def test_high_trust(self, source_tiers):
        metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)
        test_case = LLMTestCase(
            input="What is the revenue?",
            actual_output="The revenue is 100M.",
            retrieval_context=["According to the SEC filings, the revenue is 100M."]
        )
        metric.measure(test_case)
        assert metric.score == 1.0
        assert metric.success is True
        assert "Matched source 'SEC' mapped to Tier 1" in metric.reason

    def test_low_trust(self, source_tiers):
        metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)
        test_case = LLMTestCase(
            input="What is the revenue?",
            actual_output="The revenue is 100M.",
            retrieval_context=["I read on a Blog that the revenue is 100M."]
        )
        metric.measure(test_case)
        assert metric.score == 0.4
        assert metric.success is False
        assert "Matched source 'Blog' mapped to Tier 4" in metric.reason

    def test_mixed_trust(self, source_tiers):
        metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)
        test_case = LLMTestCase(
            input="What is the revenue?",
            actual_output="The revenue is 100M.",
            retrieval_context=[
                "According to the SEC filings, the revenue is 100M.",
                "Also, some Reddit thread said it was 100M."
            ]
        )
        metric.measure(test_case)
        # T1 (1.0) + T5 (0.2) = 1.2 / 2 = 0.6
        assert metric.score == 0.6
        assert metric.success is False
        assert "Tier 1" in metric.reason
        assert "Tier 5" in metric.reason

    def test_unmatched_trust(self, source_tiers):
        metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)
        test_case = LLMTestCase(
            input="What is the revenue?",
            actual_output="The revenue is 100M.",
            retrieval_context=["According to an unknown source, the revenue is 100M."]
        )
        metric.measure(test_case)
        assert metric.score == 0.5
        assert metric.success is False
        assert "Unmatched source" in metric.reason

    def test_empty_retrieval_context(self, source_tiers):
        metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)
        test_case = LLMTestCase(
            input="What is the revenue?",
            actual_output="The revenue is 100M.",
            retrieval_context=[]
        )
        metric.measure(test_case)
        assert metric.score == 0.5
        assert metric.success is False
        assert "Empty retrieval context" in metric.reason

    @pytest.mark.asyncio
    async def test_async_measure(self, source_tiers):
        metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)
        test_case = LLMTestCase(
            input="What is the revenue?",
            actual_output="The revenue is 100M.",
            retrieval_context=["According to Bloomberg, the revenue is 100M."]
        )
        score = await metric.a_measure(test_case)
        assert score == 0.8
        assert metric.score == 0.8
        assert metric.success is True
