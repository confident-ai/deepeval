import pytest
import openai
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import LLMEvalMetric
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


# # set openai api type
# openai.api_type = "azure"

# # The azure openai version you want to use
# openai.api_version = "2023-03-15"

# # The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
# openai.api_base = "https://your-resource-name.openai.azure.com/"
# openai.api_key = "<your Azure OpenAI API key>"

# def test_azure_openai_chat_completion():
#     """Test Chat Completion"""
#     metric = LLMEvalMetric(
#         name="Validity",
#         criteria="The response is a valid response to the prompt.",
#         minimum_score=0.5,
#         evaluation_params=[
#             LLMTestCaseParams.INPUT,
#             LLMTestCaseParams.ACTUAL_OUTPUT,
#         ],
#         deployment_id="your-deployment-id",
#     )
#     test_case = LLMTestCase(
#         input="What is the capital of France?",
#         actual_output="Paris",
#         expected_output="Paris",
#         context=["Geography"],
#     )

#     assert_test(test_case, [metric])
