import asyncio
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent

instrument_pydantic_ai()
agent = Agent(
    "openai:gpt-4o-mini", system_prompt="Be concise, reply with one sentence."
)
answer_relavancy_metric = AnswerRelevancyMetric()

from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's 7 * 8?"),
        Golden(input="What's 7 * 6?"),
    ]
)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        agent.run(
            golden.input,
            metrics=[answer_relavancy_metric],
        )
    )
    dataset.evaluate(task)
