from pydantic_ai import Agent
from deepeval.tracing import observe
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai

instrument_pydantic_ai()


@observe(type="tool", metric_collection="test_collection_1")
def get_weather(city: str) -> str:
    """Gets the weather for a given city."""
    return f"I don't know the weather for {city}."


agent = Agent(
    "openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful weather agent.",
)
result = agent.run_sync(
    "What is the weather in London?",
)
