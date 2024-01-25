import pytest
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator,
    DeepEvalSummarizationEvaluator,
    DeepEvalBiasEvaluator,
    DeepEvalToxicityEvaluator,
)


def test_answer_relevancy():
    evaluator = DeepEvalAnswerRelevancyEvaluator()
    assert evaluator is not None


def test_faithfulness():
    evaluator = DeepEvalFaithfulnessEvaluator()
    assert evaluator is not None


def test_contextual_relevancy():
    evaluator = DeepEvalContextualRelevancyEvaluator()
    assert evaluator is not None


def test_summarization():
    evaluator = DeepEvalSummarizationEvaluator()
    assert evaluator is not None


def test_bias():
    evaluator = DeepEvalBiasEvaluator()
    assert evaluator is not None


def test_toxicity():
    evaluator = DeepEvalToxicityEvaluator()
    assert evaluator is not None
