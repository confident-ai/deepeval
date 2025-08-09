from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain.callback import CallbackHandler
import asyncio
import time

from dotenv import load_dotenv

load_dotenv()


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)


async def run_concurrent_invokes():
    # Define 3 different inputs for concurrent execution
    inputs = [
        {
            "messages": [
                {"role": "user", "content": "what is the weather in sf"}
            ]
        },
        # {
        #     "messages": [
        #         {"role": "user", "content": "what is the weather in nyc"}
        #     ]
        # },
        # {
        #     "messages": [
        #         {"role": "user", "content": "what is the weather in la"}
        #     ]
        # },
    ]

    # Create tasks for concurrent execution
    tasks = [
        agent.ainvoke(
            input=input_data,
            config={
                "callbacks": [
                    CallbackHandler(
                        name="langgraph-test",
                        tags=["langgraph", "test"],
                        metadata={"environment": "test"},
                        thread_id="123",
                        user_id="456",
                    )
                ]
            },
        )
        for input_data in inputs
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    return results


# Run the concurrent invokes
if __name__ == "__main__":
    asyncio.run(run_concurrent_invokes())
    time.sleep(10)
