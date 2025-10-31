from openai import OpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext
from tests.test_integrations.utils import assert_trace_json, generate_trace_json
import os

client = OpenAI()

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
dataset = EvaluationDataset(goldens=goldens)

_current_dir = os.path.dirname(os.path.abspath(__file__))


# @generate_trace_json(
#     json_path=os.path.join(_current_dir, "test_response_sync_eval.json"),
#     is_run=True
# )
@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_response_sync_eval.json"),
    is_run=True
)
def test_response_sync_eval():
    for golden in dataset.evals_iterator():
        # run OpenAI client
        with trace(
            llm_span_context=LlmSpanContext(
                metrics=[AnswerRelevancyMetric(), BiasMetric()],
                expected_output=golden.expected_output,
            )
        ):
            client.responses.create(
                model="gpt-4o",
                instructions="You are a helpful assistant.",
                input=golden.input,
            )