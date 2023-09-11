import os

import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.factual_consistency import (
    FactualConsistencyMetric,
    assert_factual_consistency,
)
from deepeval.run_test import run_test

IMPLEMENTATION_NAME = "Test Factual Consistency"
os.environ["CONFIDENT_AI_IMP_NAME"] = IMPLEMENTATION_NAME


def test_factual_consistency():
    with pytest.raises(AssertionError):
        assert_factual_consistency(
            context="After a long day at work, Sarah decided to go for a walk in the park to unwind. She put on her sneakers and grabbed her headphones before heading out. As she strolled along the path, she noticed families having picnics, children playing on the playground, and ducks swimming in the pond.",
            output="Sarah spent the evening at the library, engrossed in a book.",
        )


def test_factual_consistency_2():
    assert_factual_consistency(
        context="After a long day at work, Sarah decided to go for a walk in the park to unwind. She put on her sneakers and grabbed her headphones before heading out. As she strolled along the path, she noticed families having picnics, children playing on the playground, and ducks swimming in the pond.",
        output="Sarah went out for a walk in the park.",
    )


def test_factual_consistency_metric():
    test_case = LLMTestCase(
        output="Python is a programming language.",
        context="Python is a high-level, versatile, and interpreted programming language known for its simplicity and readability.",
        metrics=[metric],
    )
    metric = FactualConsistencyMetric(minimum_score=0.8)
    run_test(test_case, [metric])


def test_factual_consistency_metric_2():
    metric = FactualConsistencyMetric(minimum_score=0.6)
    test_case = LLMTestCase(
        output="Python is a programming language.",
        context="Python is NOT a programming language.",
    )
    result = metric.measure(test_case)
    with pytest.raises(AssertionError):
        run_test(test_case, [metric])


def test_factual_consistency_metric_3():
    metric = FactualConsistencyMetric(minimum_score=0.6)
    test_case = LLMTestCase(
        output="Python is a programming language.",
        context="Python is a snake.",
    )
    with pytest.raises(AssertionError):
        run_test(test_case, metric)
