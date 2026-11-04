import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import RetrievalRedundancyMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData


def _test_case(retrieval_context):
    return LLMTestCase(
        input="Tell me about cats.",
        actual_output="Cats are great.",
        retrieval_context=retrieval_context,
    )


class TestRetrievalRedundancyMetric:
    """RetrievalRedundancyMetric is deterministic, no API key needed."""

    def test_distinct_passages_pass(self):
        metric = RetrievalRedundancyMetric()
        metric.measure(
            _test_case(
                [
                    "The cat sat on the mat.",
                    "Quantum mechanics explains particles.",
                    "Paris is the capital of France.",
                ]
            )
        )
        assert metric.score == 1.0
        assert metric.score_breakdown["overlap_ratio"] == 0.0
        assert metric.is_successful() is True

    def test_identical_passages_fail(self):
        metric = RetrievalRedundancyMetric(threshold=0.5)
        metric.measure(
            _test_case(
                [
                    "The cat sat on the mat.",
                    "The cat sat on the mat.",
                ]
            )
        )
        assert metric.score == 0.0
        assert metric.score_breakdown["redundant_pairs"] == 1.0
        assert metric.is_successful() is False

    def test_partial_redundancy(self):
        metric = RetrievalRedundancyMetric(threshold=0.5)
        metric.measure(
            _test_case(
                [
                    "The cat sat on the mat.",
                    "The cat sat on the mat.",
                    "Paris is the capital of France.",
                ]
            )
        )
        assert metric.score == pytest.approx(1 - 1 / 3)
        assert metric.score_breakdown["total_pairs"] == 3.0

    def test_single_passage_passes(self):
        metric = RetrievalRedundancyMetric()
        metric.measure(_test_case(["Only one passage."]))
        assert metric.score == 1.0

    def test_retrieved_context_data_supported(self):
        metric = RetrievalRedundancyMetric()
        metric.measure(
            _test_case(
                [
                    RetrievedContextData(
                        context="The cat sat on the mat.",
                        source="a",
                    ),
                    RetrievedContextData(
                        context="Dogs bark at night.",
                        source="b",
                    ),
                ]
            )
        )
        assert metric.score == 1.0

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            RetrievalRedundancyMetric(similarity_threshold=0)

    def test_invalid_ngram_rejected(self):
        with pytest.raises(ValueError):
            RetrievalRedundancyMetric(ngram_n=0)

    def test_missing_retrieval_context_raises(self):
        metric = RetrievalRedundancyMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(LLMTestCase(input="Tell me.", actual_output="Cats."))

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = RetrievalRedundancyMetric()
        score = await metric.a_measure(
            _test_case(["Same passage here.", "Same passage here."])
        )
        assert score == 0.0
