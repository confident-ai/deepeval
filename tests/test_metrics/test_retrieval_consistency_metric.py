"""Tests for RetrievalConsistencyMetric (no LLM involved)."""

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import RetrievalConsistencyMetric
from deepeval.test_case import LLMTestCase


def case(runs):
    return LLMTestCase(
        input="q", actual_output="a", metadata={"retrieval_runs": runs}
    )


def test_id_mode_jaccard_ignores_order():
    metric = RetrievalConsistencyMetric(mode="id", max_comparisons=3)
    metric.measure(case([["a", "b", "c"], ["c", "b", "a"], ["a", "b", "d"]]))
    # pairs: (1,2)=1.0 ; the two pairs with run 3 have |inter|=2, |union|=4 -> 0.5
    assert metric.score == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert metric.is_successful() is False


def test_ngram_mode_survives_rechunking_where_ids_do_not():
    text = (
        "backup jobs run nightly and retention is thirty days "
        "for all gold tier policies"
    )
    words = text.split()
    rechunked = [" ".join(words[:7]), " ".join(words[6:])]
    runs = [[text], rechunked, [text]]
    ngram = RetrievalConsistencyMetric(mode="ngram", ngram_size=2)
    ngram.measure(case(runs))
    ids = RetrievalConsistencyMetric(mode="id")
    ids.measure(case(runs))
    assert ngram.score > 0.9
    assert ids.score < ngram.score


def test_identical_runs_and_reason():
    metric = RetrievalConsistencyMetric(mode="id")
    metric.measure(case([["a", "b"]] * 3))
    assert metric.score == 1.0 and metric.is_successful()
    assert "chunk-ID overlap" in metric.reason


def test_needs_multiple_runs_and_valid_mode():
    metric = RetrievalConsistencyMetric(mode="id")
    with pytest.raises(MissingTestCaseParamsError, match="at least 2 runs"):
        metric.measure(case([["a"]]))
    with pytest.raises(ValueError):
        RetrievalConsistencyMetric(mode="bogus")


@pytest.mark.asyncio
async def test_a_measure_matches_measure():
    metric = RetrievalConsistencyMetric(mode="id")
    score = await metric.a_measure(case([["a"], ["b"]]))
    assert score == 0.0
