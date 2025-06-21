import os
import asyncio
import time

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import llama_index.core.instrumentation as instrument

import deepeval
from deepeval.integrations import LLamaIndexEventHandler
from deepeval.tracing import observe

os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"

deepeval.login_with_confident_api_key("<your-confident-api-key>")

dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(LLamaIndexEventHandler())

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)

@observe()
async def main():   
    response = await agent.run(
        "What's 7 * 8?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(12)
