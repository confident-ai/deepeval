from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
from langgraph.prebuilt import create_react_agent
from deepeval.evaluate import dataset
from deepeval.dataset import Golden


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

for golden in dataset(goldens=[Golden(input="This is a test query")]):
    agent.invoke(
        input={"messages": [{"role": "user", "content": golden.input}]},
        config={
            "callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]
        },
    )
