import os
import asyncio
import time

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import llama_index.core.instrumentation as instrument

import deepeval
from deepeval.integrations import instrument_llama_index

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

deepeval.login_with_confident_api_key("<YOUR_CONFIDENT_API_KEY>")

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
    response = await agent.run(
        "What's 7 * 8?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(12)
