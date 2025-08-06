from deepeval.integrations.llama_index import (
    instrument_llama_index,
    FunctionAgent,
)
import llama_index.core.instrumentation as instrument
from llama_index.llms.openai import OpenAI
from deepeval.evaluate import dataset, test_run
from deepeval.dataset import Golden
import asyncio
from deepeval.metrics import AnswerRelevancyMetric

instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    return a * b


metric = AnswerRelevancyMetric()
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[metric],
)


def main():
    async def llm_app(input_text):
        response = await agent.run(input_text)
        return response

    for golden in dataset(goldens=[Golden(input="What is 1234 * 4567?")]):
        task = asyncio.create_task(llm_app(golden.input))
        test_run.append(task)


main()
