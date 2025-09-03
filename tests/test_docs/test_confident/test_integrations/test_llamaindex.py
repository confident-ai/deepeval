from deepeval.integrations.llama_index import FunctionAgent
from llama_index.llms.openai import OpenAI
import llama_index.core.instrumentation as instrument
import asyncio

from deepeval.integrations.llama_index import instrument_llama_index

# Don't forget to setup tracing
instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metric_collection="My Metrics",
)


async def main():
    response = await agent.run("What's 7 * 8?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
