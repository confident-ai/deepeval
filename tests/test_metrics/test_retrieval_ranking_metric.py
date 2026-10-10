import math

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import RetrievalRankingMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData


def _test_case(retrieval_context, context):
    return LLMTestCase(
        input="What is the Eiffel Tower?",
        actual_output="The Eiffel Tower is tall.",
        retrieval_context=retrieval_context,
        context=context,
    )


class TestRetrievalRankingMetric:
    """Deterministic metric, so these tests run without an API key."""

    def test_perfect_ranking_scores_one(self):
        metric = RetrievalRankingMetric(k=3)
        metric.measure(
            _test_case(
                ["chunk A", "chunk B", "chunk C"],
                ["chunk A", "chunk B"],
            )
        )
        assert metric.score == 1.0
        assert metric.score_breakdown["recall@k"] == 1.0
        assert metric.score_breakdown["precision@k"] == pytest.approx(2 / 3)
        assert metric.score_breakdown["mrr"] == 1.0
        assert metric.score_breakdown["ndcg"] == pytest.approx(1.0)
        assert metric.score_breakdown["hit_rate"] == 1.0
        assert metric.is_successful() is True

    def test_mrr_uses_first_relevant_rank(self):
        metric = RetrievalRankingMetric(k=3, metric="mrr")
        metric.measure(
            _test_case(
                ["irrelevant", "irrelevant", "target chunk"],
                ["target chunk"],
            )
        )
        assert metric.score == pytest.approx(1 / 3)
        assert metric.score_breakdown["recall@k"] == 1.0
        assert metric.score_breakdown["hit_rate"] == 1.0

    def test_ndcg_discounts_lower_ranks(self):
        metric = RetrievalRankingMetric(k=3, metric="ndcg")
        metric.measure(
            _test_case(
                ["noise", "relevant one", "relevant two"],
                ["relevant one", "relevant two"],
            )
        )
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert metric.score == pytest.approx(dcg / idcg)
        assert metric.score_breakdown["precision@k"] == pytest.approx(2 / 3)

    def test_precision_and_hit_rate(self):
        metric = RetrievalRankingMetric(k=4, metric="precision")
        metric.measure(
            _test_case(
                ["keep", "drop", "drop", "keep"],
                ["keep"],
            )
        )
        assert metric.score == pytest.approx(0.5)
        assert metric.score_breakdown["hit_rate"] == 1.0
        assert metric.score_breakdown["recall@k"] == 1.0

    def test_no_hit_scores_zero(self):
        metric = RetrievalRankingMetric(k=2)
        metric.measure(_test_case(["a", "b"], ["zzz"]))
        assert metric.score == 0.0
        assert metric.score_breakdown["mrr"] == 0.0
        assert metric.score_breakdown["ndcg"] == 0.0
        assert metric.score_breakdown["hit_rate"] == 0.0
        assert metric.is_successful() is False

    def test_k_truncates_ranking(self):
        metric = RetrievalRankingMetric(k=1)
        metric.measure(
            _test_case(["noise", "wanted"], ["wanted"]),
        )
        assert metric.score == 0.0

        full = RetrievalRankingMetric()
        full.measure(_test_case(["noise", "wanted"], ["wanted"]))
        assert full.score == 1.0
        assert full.score_breakdown["mrr"] == pytest.approx(0.5)

    def test_contains_matching_and_retrieved_context_data(self):
        metric = RetrievalRankingMetric(k=2)
        metric.measure(
            _test_case(
                [
                    RetrievedContextData(
                        context="The Eiffel Tower is 330 metres tall.",
                        source="tower",
                    ),
                    "unrelated passage",
                ],
                ["330 metres tall"],
            )
        )
        assert metric.score == 1.0

    def test_exact_match_mode_is_strict(self):
        metric = RetrievalRankingMetric(k=2, match_mode="exact")
        metric.measure(
            _test_case(
                ["The Eiffel Tower is 330 metres tall."],
                ["330 metres tall"],
            )
        )
        assert metric.score == 0.0

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValueError):
            RetrievalRankingMetric(metric="bogus")

    def test_non_positive_k_rejected(self):
        with pytest.raises(ValueError):
            RetrievalRankingMetric(k=0)

    def test_missing_context_raises(self):
        metric = RetrievalRankingMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(
                LLMTestCase(
                    input="q",
                    actual_output="a",
                    retrieval_context=["a"],
                )
            )

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = RetrievalRankingMetric(k=2, metric="hit_rate")
        score = await metric.a_measure(_test_case(["keep", "drop"], ["keep"]))
        assert score == 1.0
