import asyncio

from llama_index.llms.openai import OpenAI
import llama_index.core.instrumentation as instrument

from deepeval.integrations.llama_index import (
    instrument_llama_index,
    FunctionAgent,
)

instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metric_collection="test_collection_1",
)


async def llm_app(input: str):
    return await agent.run(input)


def execute_agent():
    return asyncio.run(llm_app("What is 3 * 12?"))
