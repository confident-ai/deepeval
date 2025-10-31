import asyncio
import os
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from tests.test_integrations.utils import assert_trace_json, generate_trace_json

agent = Agent(
    "openai:gpt-5",
    instructions="You are a helpful assistant.",
    instrument=ConfidentInstrumentationSettings(
        is_test_mode=True, agent_metrics=[AnswerRelevancyMetric()]
    ),
)


async def run_agent(input: str):
    return await agent.run(input)


dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's the weather in Paris?"),
        Golden(input="What's the weather in London?"),
    ]
)

_current_dir = os.path.dirname(os.path.abspath(__file__))


# @generate_trace_json(
#     json_path=os.path.join(_current_dir, "test_async_eval.json"),
#     is_run=True
# )
@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_async_eval.json"), is_run=True
)
def test_async_eval():
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(run_agent(golden.input))
        dataset.evaluate(task)
