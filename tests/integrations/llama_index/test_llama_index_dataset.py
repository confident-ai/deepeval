import os
from deepeval.integrations.llama_index import instrument_llama_index
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio
import time

import deepeval
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from dotenv import load_dotenv

load_dotenv()


# Don't forget to setup tracing
deepeval.login(os.getenv("CONFIDENT_API_KEY"))
instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


answer_relevancy_metric = AnswerRelevancyMetric()
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    # metric_collection="test_collection_1",
    metrics=[answer_relevancy_metric],
)

goldens = [Golden(input="What's 7 * 8?"), Golden(input="What's 7 * 6?")]


async def llm_app(golden: Golden):
    await agent.run(golden.input)


def main():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(llm_app(golden))
        dataset.evaluate(task)


if __name__ == "__main__":
    main()
    time.sleep(7)
