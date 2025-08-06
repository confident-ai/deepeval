import os
import time
import asyncio
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import Agent as DeepEvalAgent
from dotenv import load_dotenv

load_dotenv()

instrument_pydantic_ai(api_key=os.getenv("CONFIDENT_API_KEY"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Agent.instrument_all()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
)

deep_eval_agent = DeepEvalAgent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    metric_collection="test_collection_1",
)


async def run_concurrent_agents():
    # Define 3 different questions for concurrent execution
    questions = [
        'Where does "hello world" come from?',
        "What is the capital of France?",
        "How many planets are in our solar system?",
    ]

    # Create tasks for concurrent execution
    tasks = [agent.run(question) for question in questions]
    deep_eval_tasks = [deep_eval_agent.run(question) for question in questions]

    # Run all tasks concurrently and wait for all to complete
    results = await asyncio.gather(*tasks)
    deep_eval_results = await asyncio.gather(*deep_eval_tasks)
    # Print all results
    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result.output}")

    for i, result in enumerate(deep_eval_results, 1):
        print(f"Deep Eval Result {i}: {result.output}")

    return results


# Run the concurrent function
if __name__ == "__main__":
    asyncio.run(run_concurrent_agents())

    time.sleep(10)
