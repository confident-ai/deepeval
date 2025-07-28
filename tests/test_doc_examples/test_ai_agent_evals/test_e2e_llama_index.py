from deepeval.integrations.llama_index import instrument_llama_index, FunctionAgent
import llama_index.core.instrumentation as instrument
from deepeval.metrics import TaskCompletionMetric
from llama_index.llms.openai import OpenAI
from deepeval.evaluate import dataset, test_run
from deepeval.dataset import Golden
import asyncio

instrument_llama_index(instrument.get_dispatcher())

def multiply(a: float, b: float) -> float:
    return a * b

agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metrics=[TaskCompletionMetric()]
)

async def llm_app(input: str):   
    response = await agent.run(input)
    return response

async def main():
    for golden in dataset(goldens=[Golden(input="What is 1234 * 4567?")]):
        tasks = asyncio.create_task(llm_app(golden.input))
        test_run.append(tasks)

if __name__ == "__main__":
    asyncio.run(main())