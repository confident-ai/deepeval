"""Tests for answer relevancy
"""
import pytest

query = "What is Python?"
answer = "Python is a programming language?"


@pytest.mark.asyncio
async def test_answer_relevancy():
    from deepeval.test_utils import assert_answer_relevancy

    assert_answer_relevancy(query, answer, success_threshold=0.5)


@pytest.mark.asyncio
async def test_query_answer_relevancy():
    from deepeval.metrics.answer_relevancy import AnswerRelevancy

    scorer = AnswerRelevancy(success_threshold=0.5)
    result = scorer.measure(query=query, answer=answer)
