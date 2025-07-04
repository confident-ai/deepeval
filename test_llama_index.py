import os
import asyncio
import time
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from deepeval.integrations.llama_index import instrumentator

os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"
instrumentator(api_key="<your_deepeval_api_key>")

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
