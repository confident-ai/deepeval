import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import LLMEvalMetric
from deepeval import assert_test


def test_chat_completion():
    """Test Chat Completion"""
    metric = LLMEvalMetric(
        name="NER",
        criteria="The response is a valid response to the prompt.",
        minimum_score=0.5,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    test_case = LLMTestCase(
        input="Do the NER of the following sentence: George H.W. Bush visited China in 1989.",
        actual_output=["PER","LOC"],
        expected_output=["PER","LOC"],
    )

    assert_test(test_case, [metric])
