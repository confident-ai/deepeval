import asyncio

from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent
from deepeval.metrics import AnswerRelevancyMetric

instrument_pydantic_ai()

Agent.instrument_all()

answer_relavancy_metric = AnswerRelevancyMetric()
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    metrics=[answer_relavancy_metric]
)

from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(goldens=[
    Golden(input="What's 7 * 8?"),
    Golden(input="What's 7 * 6?"),
])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(agent.run(golden.input))
    dataset.evaluate(task)