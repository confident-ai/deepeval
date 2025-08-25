from deepeval.metrics import GEval
from deepeval.tracing import observe, update_current_trace
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import ToolCall
from deepeval.dataset import EvaluationDataset


relevnacy = GEval(
    name="Relevancy",
    criteria="For the given input, the output should be relevant to the input.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)
correctness = GEval(
    name="Correctness",
    criteria="Given the expected output, determine whether the output is correct or not.",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)


@observe()
def llm_app(input):
    update_current_trace(
        input=input,
        output="Hi",
        expected_output="Hi",
        retrieval_context=["Hi"],
        context=["Hi"],
        tools_called=[ToolCall(name="Hi")],
        expected_tools=[ToolCall(name="Hi")],
    )
    return "Hi"


dataset = EvaluationDataset()
dataset.pull(alias="New Dataset")

for golden in dataset.evals_iterator(metrics=[relevnacy, correctness]):
    llm_app(golden.input)
