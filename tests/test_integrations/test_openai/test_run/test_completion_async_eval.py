import os
import asyncio
from openai import AsyncOpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext
from tests.test_integrations.utils import assert_trace_json, generate_trace_json

async_client = AsyncOpenAI()

async def openai_llm_call(input):
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
            # expected_output=golden.expected_output,
        )
    ):
        return await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot. Always generate a string response."},
                {"role": "user", "content": input},
            ],
        )

_current_dir = os.path.dirname(os.path.abspath(__file__))


# @generate_trace_json(
#     json_path=os.path.join(_current_dir, "test_completion_async_eval.json"),
#     is_run=True
# )
@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_completion_async_eval.json"),
    is_run=True
)
def test_completion_async_eval():
    goldens = [
        Golden(input="What is the weather in Bogotá, Colombia?"),
        Golden(input="What is the weather in Paris, France?"),
    ]
    dataset = EvaluationDataset(goldens=goldens)

    for golden in dataset.evals_iterator():
        task = asyncio.create_task(openai_llm_call(golden.input))
        dataset.evaluate(task)

# if __name__ == "__main__":
#     test_completion_async_eval()