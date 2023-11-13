import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.llm_eval_metric import LLMEvalMetric
from deepeval.evaluator import assert_test


def test_chat_completion():
    """Test Chat Completion"""
    metric = LLMEvalMetric(
        name="Validity",
        criteria="The response is a valid response to the prompt.",
        minimum_score=0.5,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris",
        context=["Geography"],
    )

    assert_test(test_case, [metric])
