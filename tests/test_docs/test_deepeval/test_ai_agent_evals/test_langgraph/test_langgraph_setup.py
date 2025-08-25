from langgraph.prebuilt import create_react_agent
from deepeval.metrics import TaskCompletionMetric

from deepeval.integrations.langchain import CallbackHandler


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


task_completion_metric = TaskCompletionMetric()

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

# run for testing (not needed for docs)
result = agent.invoke(
    input={
        "messages": [{"role": "user", "content": "what is the weather in sf"}]
    },
    config={"callbacks": [CallbackHandler()]},
)

print(result)
