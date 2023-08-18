"""This is the ideal user flow
"""
import pytest
from deepeval.test_utils import assert_llm_output
from deepeval.metrics.bertscore_metric import BertScoreMetric


def generate_llm_output(input: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output


@pytest.mark.asyncio
def test_llm_output():
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    assert_llm_output(output, expected_output, metric="entailment")
    assert_llm_output(output, expected_output, metric="bertscore")


@pytest.mark.asyncio
def test_llm_output_custom():
    input = "What is the customer success phone line?"
    expected_output = "Dogs and cats love to walk around the beach."
    output = generate_llm_output(input)
    metric = BertScoreMetric(minimum_score=0.98)
    with pytest.raises(AssertionError):
        assert_llm_output(output, expected_output, metric=metric)
