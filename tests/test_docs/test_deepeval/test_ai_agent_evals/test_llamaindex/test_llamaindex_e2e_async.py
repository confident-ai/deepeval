import asyncio

from llama_index.llms.openai import OpenAI
import llama_index.core.instrumentation as instrument

from deepeval.integrations.llama_index import (
    instrument_llama_index,
    FunctionAgent,
)

from deepeval.metrics import AnswerRelevancyMetric

instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


answer_relevancy_metric = AnswerRelevancyMetric()

agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[answer_relevancy_metric],
)


async def llm_app(input: str):
    return await agent.run(input)


from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(
    goldens=[Golden(input="What is 3 * 12?"), Golden(input="What is 4 * 13?")]
)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(golden.input))
    dataset.evaluate(task)
