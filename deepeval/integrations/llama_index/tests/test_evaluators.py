import pytest
from deepeval.integrations.llama_index import (
    AnswerRelevancyEvaluator,
    FaithfulnessEvaluator,
    ContextualRelevancyEvaluator,
    SummarizationEvaluator,
    BiasEvaluator,
    ToxicityEvaluator,
)


def test_answer_relevancy():
    evaluator = AnswerRelevancyEvaluator()
    assert evaluator is not None


def test_faithfulness():
    evaluator = FaithfulnessEvaluator()
    assert evaluator is not None


def test_contextual_relevancy():
    evaluator = ContextualRelevancyEvaluator()
    assert evaluator is not None


def test_summarization():
    evaluator = SummarizationEvaluator()
    assert evaluator is not None


def test_bias():
    evaluator = BiasEvaluator()
    assert evaluator is not None


def test_toxicity():
    evaluator = ToxicityEvaluator()
    assert evaluator is not None
