"""This is the ideal user flow
"""
import pytest

from deepeval.metrics import HallucinationMetric

from deepeval.test_case import LLMTestCase
from deepeval.evaluator import assert_test


def generate_llm_output(query: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output


def test_llm_output():
    input = "What is the customer success phone line?"
    context = ["Our customer success phone line is 1200-231-231."]
    test_case = LLMTestCase(
        input=input, actual_output=generate_llm_output(input), context=context
    )
    assert_test(test_case, [HallucinationMetric(minimum_score=0.5)])


def test_llm_output_custom():
    actual_output = "Dogs and cats hate to walk around the beach."
    context = ["Dogs and cats love to walk around the beach."]
    test_case = LLMTestCase(
        input="Placerholder", actual_output=actual_output, context=context
    )
    with pytest.raises(AssertionError):
        assert_test(test_case, [HallucinationMetric(minimum_score=0.5)])


def test_0():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = ["Biology"]

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected_output,
        context=context,
    )
    metric = HallucinationMetric()
    with pytest.raises(AssertionError):
        assert_test(test_case, metrics=[metric])
