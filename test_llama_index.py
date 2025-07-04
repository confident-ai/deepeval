import os
import asyncio
import time
import deepeval
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from deepeval.integrations.llama_index import instrument_llama_index
import llama_index.core.instrumentation as instrument

os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"
deepeval.login_with_confident_api_key("<your_deepeval_api_key>")

instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)


async def main():
    response = await agent.run("What's 7 * 8?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(8)
