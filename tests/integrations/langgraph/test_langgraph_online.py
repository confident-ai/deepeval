import os
import time
from langgraph.prebuilt import create_react_agent

from deepeval.integrations.langchain.callback import CallbackHandler
import deepeval
from deepeval.metrics import TaskCompletionMetric


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

result = agent.invoke(
    input={
        "messages": [{"role": "user", "content": "what is the weather in sf"}]
    },
    config={
        "callbacks": [
            CallbackHandler(metric_collection="Task Completion Collection")
        ]
    },
)

time.sleep(5)  # Wait for the trace to be published
