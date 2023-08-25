"""Tests for answer relevancy
"""
import pytest
from deepeval.metrics.answer_relevancy import assert_answer_relevancy
from deepeval.metrics.answer_relevancy import AnswerRelevancy

query = "What is Python?"
answer = "Python is a programming language?"


@pytest.mark.asyncio
async def test_answer_relevancy():

    assert_answer_relevancy(query, answer, minimum_score=0.5)


@pytest.mark.asyncio
async def test_answer_not_relevant():
    with pytest.raises(AssertionError):
        assert_answer_relevancy(query, "He is not your friend")


@pytest.mark.asyncio
async def test_query_answer_relevancy():

    scorer = AnswerRelevancy(minimum_score=0.5)
    result = scorer.measure(query=query, output=answer)
    assert scorer.is_successful()


@pytest.mark.asyncio
async def test_compare_answer_relevancy():
    scorer = AnswerRelevancy(minimum_score=0.5)
    result = scorer.measure(query=query, output="Programming lang")
    result_2 = scorer.measure(query=query, output="Python is a programming lang")
    assert result_2 > result
