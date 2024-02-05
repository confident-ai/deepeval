import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import assert_test


@pytest.mark.skip(reason="openai is expensive")
def test_g_eval():
    metric = GEval(
        name="Validity",
        criteria="The response is a valid response to the prompt.",
        threshold=0.5,
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
