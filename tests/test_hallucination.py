import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from deepeval import assert_test
import deepeval


@pytest.mark.skip(reason="openai is expensive")
def test_hallucination_metric():
    metric = HallucinationMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="A blond drinking water in public.",
        context=[
            "A man with blond-hair, and a brown shirt drinking out of a public water fountain."
        ],
        cost=0.4,
        latency=2,
    )
    assert_test(test_case, [metric])


@pytest.mark.skip(reason="openai is expensive")
def test_hallucination_metric_2():
    metric = HallucinationMetric(threshold=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is NOT a programming language."],
        cost=1,
        latency=0.2,
    )
    assert_test(test_case, [metric])


@pytest.mark.skip(reason="openai is expensive")
def test_hallucination_metric_3():
    metric = HallucinationMetric(threshold=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is a snake.", "Pythons like to lurk in the forests."],
        cost=0.1,
        latency=13.0,
    )
    assert_test(test_case, [metric])


# prompt_template = """You are a helpful assistant, answer the following question without using any prior knowledge.

# Question:
# {question}
# """


# @deepeval.log_hyperparameters(model="gpt-4", prompt_template=prompt_template)
# def hyperparameters():
#     return {
#         "model": "gpt-4",
#     }
