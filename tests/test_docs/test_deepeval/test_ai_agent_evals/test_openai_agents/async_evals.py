import asyncio
from agents import Runner, add_trace_processor
from deepeval.openai_agents import Agent, DeepEvalTracingProcessor
from deepeval.metrics import AnswerRelevancyMetric

add_trace_processor(DeepEvalTracingProcessor())

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


if __name__ == "__main__":
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(Runner.run(weather_agent, golden.input))
        dataset.evaluate(task)