import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import ChunkUtilizationMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData

PASSAGES = [
    "The Eiffel Tower is 330 metres tall.",
    "The Eiffel Tower was completed in 1889.",
    "Paris is the capital of France.",
]


def _test_case(actual_output, retrieval_context=None):
    return LLMTestCase(
        input="Tell me about the Eiffel Tower.",
        actual_output=actual_output,
        retrieval_context=retrieval_context or PASSAGES,
    )


class TestChunkUtilizationMetric:
    """ChunkUtilizationMetric is deterministic, no API key needed."""

    def test_all_chunks_used_passes(self):
        metric = ChunkUtilizationMetric()
        metric.measure(_test_case("Tall [1]. Old [2]. Paris [3]."))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_partial_use(self):
        metric = ChunkUtilizationMetric(threshold=0.5)
        metric.measure(_test_case("Tall [1]. Very popular."))
        assert metric.score == pytest.approx(1 / 3)
        assert metric.is_successful() is False

    def test_no_citations_scores_zero(self):
        metric = ChunkUtilizationMetric()
        metric.measure(_test_case("The tower is tall."))
        assert metric.score == 0.0
        assert "0 of 3" in metric.reason

    def test_duplicate_citations_count_once(self):
        metric = ChunkUtilizationMetric()
        metric.measure(_test_case("Tall [1]. Still tall [1]."))
        assert metric.score == pytest.approx(1 / 3)

    def test_broken_citations_not_counted(self):
        metric = ChunkUtilizationMetric()
        metric.measure(_test_case("Tall [1]. Old [7]."))
        assert metric.score == pytest.approx(1 / 3)
        assert metric.score_breakdown["broken_citations"] == 1.0

    def test_source_citations_resolve(self):
        context = [
            RetrievedContextData(context=PASSAGES[0], source="facts"),
            RetrievedContextData(context=PASSAGES[1], source="history"),
        ]
        metric = ChunkUtilizationMetric()
        metric.measure(
            _test_case("Tall [facts]. Old [2].", retrieval_context=context)
        )
        assert metric.score == 1.0

    def test_markdown_link_is_not_a_citation(self):
        metric = ChunkUtilizationMetric()
        metric.measure(_test_case("Tall [1]. See [docs](https://example.com)."))
        assert metric.score == pytest.approx(1 / 3)

    def test_strict_mode_zeroes_partial(self):
        metric = ChunkUtilizationMetric(strict_mode=True)
        metric.measure(_test_case("Tall [1]. Very popular."))
        assert metric.score == 0

    def test_missing_retrieval_context_raises(self):
        metric = ChunkUtilizationMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(
                LLMTestCase(input="Tell me.", actual_output="Tall [1].")
            )

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = ChunkUtilizationMetric()
        score = await metric.a_measure(_test_case("Tall [1]."))
        assert score == pytest.approx(1 / 3)
