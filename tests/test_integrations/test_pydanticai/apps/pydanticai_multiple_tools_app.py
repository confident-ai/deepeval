"""
PydanticAI Multiple Tools App: Agent with multiple tool definitions
Complexity: MEDIUM - Tests multiple tool calling functionality

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


def create_multiple_tools_agent(
    name: str = "pydanticai-multiple-tools-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
    tool_metric_collection_map: dict = None,
) -> Agent:
    """Create a PydanticAI agent with multiple tools and instrumentation settings."""
    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "multiple-tools"],
        metadata=metadata or {"test_type": "multiple_tools"},
        thread_id=thread_id,
        user_id=user_id,
        tool_metric_collection_map=tool_metric_collection_map or {},
        is_test_mode=True,
    )

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt=(
            "You are a helpful assistant with access to weather and time tools. "
            "When asked about weather, use the get_weather tool. "
            "When asked about time, use the get_time tool. "
            "Be concise in your responses."
        ),
        instrument=settings,
        name="multiple_tools_agent",
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """
        Get the current weather for a city.

        Args:
            city: The name of the city

        Returns:
            The current weather conditions
        """
        weather_data = {
            "tokyo": "Sunny, 72F",
            "london": "Rainy, 55F",
            "paris": "Cloudy, 62F",
            "new york": "Clear, 68F",
        }
        return weather_data.get(
            city.lower(), f"Weather data not available for {city}"
        )

    @agent.tool_plain
    def get_time(city: str) -> str:
        """
        Get the current time for a city.

        Args:
            city: The name of the city

        Returns:
            The current time in that city
        """
        time_data = {
            "tokyo": "3:00 PM JST",
            "london": "7:00 AM GMT",
            "paris": "8:00 AM CET",
            "new york": "2:00 AM EST",
        }
        return time_data.get(
            city.lower(), f"Time data not available for {city}"
        )

    return agent


def invoke_multiple_tools_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the multiple tools agent synchronously."""
    if agent is None:
        agent = create_multiple_tools_agent()
    result = agent.run_sync(prompt)
    return result.output


async def ainvoke_multiple_tools_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the multiple tools agent asynchronously."""
    if agent is None:
        agent = create_multiple_tools_agent()
    result = await agent.run(prompt)
    return result.output
