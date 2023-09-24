import pytest
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test


def test_1():
    input = "What does your company do?"
    output = "My company doesn't do anything."
    context = "Our company specializes in cloud computing, data analytics, and machine learning. We offer a range of services including cloud storage solutions, data analytics platforms, and custom machine learning models."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=1.0)
    test_case = LLMTestCase(output=output, context=context)
    with pytest.raises(AssertionError):
        assert_test(test_case, [factual_consistency_metric])


def test_2():
    input = "What does your company do?"
    output = "My company is a cloud computing company."
    context = "Our company specializes in cloud computing, data analytics, and machine learning. We offer a range of services including cloud storage solutions, data analytics platforms, and custom machine learning models."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.5)
    test_case = LLMTestCase(output=output, context=context)
    assert_test(test_case, [factual_consistency_metric])
