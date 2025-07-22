import os
from deepeval.integrations.llama_index import instrument_llama_index
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index.agent import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio
import time
 
import deepeval
 
# Don't forget to setup tracing
deepeval.login_with_confident_api_key("<confident_api_key>")
instrument_llama_index(instrument.get_dispatcher())
 
os.environ["OPENAI_API_KEY"] = "<openai_api_key>"
 
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b
 
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metric_collection="test_collection_1",
)

async def main():   
    response = await agent.run("What's 7 * 8?")
    print(response)
 
if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(7)