from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate
from deepeval.models import GPTModel, AzureOpenAIModel

# metric = GEval(
#     name="Validity",
#     criteria="The response is a valid response to the prompt",
#     threshold=0.6,
#     evaluation_params=[
#         LLMTestCaseParams.INPUT,
#         LLMTestCaseParams.ACTUAL_OUTPUT,
#     ],
#     model=AzureOpenAIModel(model="gpt-4o"),
#     async_mode=False,
# )
# test_case = LLMTestCase(
#     input="What is the capital of France?",
#     actual_output="Countries have capitals",
#     expected_output="Paris",
#     context=["Geography"],
# )

# metric.measure(test_case)
