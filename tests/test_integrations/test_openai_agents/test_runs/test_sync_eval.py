from agents import Runner, add_trace_processor
from deepeval.openai_agents import Agent, DeepEvalTracingProcessor
from deepeval.metrics import AnswerRelevancyMetric
from tests.test_integrations.utils import assert_trace_json, generate_trace_json

import os

# add_trace_processor(DeepEvalTracingProcessor())

weather_agent = Agent(
    name="Weather Agent",
    instructions="You are a weather agent. You are given a question about the weather and you need to answer it.",
    agent_metrics=[AnswerRelevancyMetric()],
)

from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's the weather in UK?"),
        Golden(input="What's the weather in France?"),
    ]
)

_current_dir = os.path.dirname(os.path.abspath(__file__))

@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_sync_evals.json"),
    is_run=True
)
def test_run_sync_evals():
    for golden in dataset.evals_iterator():
        Runner.run_sync(weather_agent, golden.input)