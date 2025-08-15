import os
import asyncio
import time

from dotenv import load_dotenv

load_dotenv()

from deepeval.integrations.pydantic_ai import instrument_pydantic_ai
from deepeval.integrations.pydantic_ai import Agent
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import Golden, EvaluationDataset


instrument_pydantic_ai(api_key=os.getenv("CONFIDENT_API_KEY"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Agent.instrument_all()

answer_relavancy_metric = AnswerRelevancyMetric()
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    metrics=[answer_relavancy_metric],
)

goldens = [Golden(input="What's 7 * 8?"), Golden(input="What's 7 * 6?")]


def main():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(agent.run(golden.input))
        dataset.evaluate(task)


if __name__ == "__main__":
    main()
    time.sleep(7)
