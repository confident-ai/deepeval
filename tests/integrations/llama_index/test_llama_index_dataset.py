import os
from deepeval.integrations.llama_index import instrument_llama_index
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio
import time

import deepeval
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import Golden
from deepeval.evaluate import dataset, test_run

# Don't forget to setup tracing
deepeval.login_with_confident_api_key("<CONFIDENT_API_KEY>")
instrument_llama_index(instrument.get_dispatcher())

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"


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
    for golden in dataset(goldens=goldens):
        task = asyncio.create_task(llm_app(golden))
        test_run.append(task)


if __name__ == "__main__":
    main()
    time.sleep(7)
