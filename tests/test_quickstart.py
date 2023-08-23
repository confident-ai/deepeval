"""This is the ideal user flow
"""
import pytest
import os
from deepeval.api import Api
from deepeval.metrics.factual_consistency import assert_factual_consistency
from deepeval.metrics.bertscore_metric import BertScoreMetric
from deepeval.constants import API_KEY_ENV

IMPLEMENTATION_NAME = "Quickstart Example"
os.environ["CONFIDENT_AI_IMP_NAME"] = IMPLEMENTATION_NAME


def generate_llm_output(input: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output


@pytest.mark.asyncio
async def test_llm_output():
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    assert_factual_consistency(output, expected_output, metric="entailment")
    assert_factual_consistency(output, expected_output, metric="bertscore")


@pytest.mark.asyncio
async def test_llm_output_custom():
    input = "What is the customer success phone line?"
    expected_output = "Dogs and cats love to walk around the beach."
    output = generate_llm_output(input)
    metric = BertScoreMetric(minimum_score=0.98)
    with pytest.raises(AssertionError):
        assert_factual_consistency(output, expected_output, metric=metric)

def test_implementation_inside_quickstart():
    api_key = os.environ[API_KEY_ENV]
    client = Api(api_key=api_key)
    imps = client.list_implementations()
    FOUND = False
    for imp in imps:
        if imp['name'] == IMPLEMENTATION_NAME:
            FOUND = True
    assert FOUND, f"Not found in {imps}"
