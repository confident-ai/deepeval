from deepeval.integrations.llama_index import (
    instrument_llama_index,
    FunctionAgent,
)
import llama_index.core.instrumentation as instrument
from llama_index.llms.openai import OpenAI
import asyncio

instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
)


async def llm_app():
    response = await agent.run("What is 1234 * 4567?")
    print(response)


asyncio.run(llm_app())
