from deepeval.integrations.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

result = agent.invoke(
    input={
        "messages": [{"role": "user", "content": "what is the weather in sf"}]
    },
    config={"callbacks": [CallbackHandler()]},
)
