import os
import time
from langgraph.prebuilt import create_react_agent

from deepeval.integrations.langchain.callback import CallbackHandler
import deepeval
from deepeval.metrics import TaskCompletionMetric

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"
deepeval.login_with_confident_api_key("<YOUR_CONFIDENT_API_KEY>")

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
    config={"callbacks": [CallbackHandler(metric_collection="task_completion")]},
)

time.sleep(5)  # Wait for the trace to be published