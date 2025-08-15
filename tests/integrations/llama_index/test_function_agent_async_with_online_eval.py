import os
from deepeval.integrations.llama_index import instrument_llama_index
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio
import time

import deepeval
from dotenv import load_dotenv

load_dotenv()

# Don't forget to setup tracing
deepeval.login(os.getenv("CONFIDENT_API_KEY"))
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


async def main():
    # Define 3 different queries
    queries = ["What's 7 * 8?", "What's 12 * 15?", "What's 23 * 4?"]

    # Run all queries concurrently using asyncio.gather
    print("Starting 3 concurrent queries...")
    start_time = time.time()

    tasks = [agent.run(query) for query in queries]
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"\nAll queries completed in {end_time - start_time:.2f}s")
    print("\nResults:")
    for response in results:
        print(f"Query: {response}")


if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(7)
