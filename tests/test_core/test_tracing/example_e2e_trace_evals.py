from deepeval.metrics.bias.bias import BiasMetric
from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall
from deepeval.evaluate.configs import AsyncConfig


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
        metrics=[AnswerRelevancyMetric(), BiasMetric()],
    )
    return "Hi"


dataset = EvaluationDataset(goldens=[Golden(input="Hello"), Golden(input="Hi")])
for golden in dataset.evals_iterator(async_config=AsyncConfig(run_async=True)):
    llm_app(golden.input)
