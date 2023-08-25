"""This is the ideal user flow
"""
import os
import pytest
from deepeval.api import Api
from deepeval.metrics.factual_consistency import assert_factual_consistency
from deepeval.metrics.bertscore_metric import BertScoreMetric
from deepeval.constants import API_KEY_ENV

IMPLEMENTATION_NAME = "Quickstart Example 2"
os.environ["CONFIDENT_AI_IMP_NAME"] = IMPLEMENTATION_NAME


def generate_llm_output(query: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output


def test_llm_output():
    query = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(query)
    assert_factual_consistency(output, expected_output)


def test_llm_output_custom():
    query = "What is the customer success phone line?"
    expected_output = "Dogs and cats love to walk around the beach."
    with pytest.raises(AssertionError):
        assert_factual_consistency(query, expected_output)


def test_implementation_inside_quickstart():
    client = Api()
    imps = client.list_implementations()
    FOUND = False
    for imp in imps:
        if imp["name"] == IMPLEMENTATION_NAME:
            FOUND = True
    assert FOUND, f"Not found in {imps}"
