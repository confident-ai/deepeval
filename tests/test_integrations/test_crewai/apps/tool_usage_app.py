"""
tests/test_integrations/test_crewai/apps/tool_usage_app.py
A crew heavily focused on tool usage to test ToolSpans.
"""

from crewai import Agent, Task, Crew, LLM
from deepeval.integrations.crewai import tool as deepeval_tool


@deepeval_tool(metric_collection="weather_tool_metrics")
def get_weather(city: str) -> str:
    """Fetch weather data for a given city."""
    # Deterministic mock response
    data = {
        "london": "Rainy, 60°F",
        "paris": "Cloudy, 65°F",
        "tokyo": "Clear, 70°F",
    }
    return f"Weather in {city}: {data.get(city.lower(), 'Sunny, 72°F')}"


def get_tool_usage_app():
    llm = LLM(model="gpt-4o-mini", temperature=0)

    agent = Agent(
        role="Meteorologist",
        goal="Check weather",
        backstory="You are a meticulous meteorologist. You NEVER guess the weather. You ALWAYS call the tool without exception.",
        tools=[get_weather],
        llm=llm,
        verbose=True,
    )

    task = Task(
        description="Check the weather in {city}. You MUST use the 'get_weather' tool to find the answer. Do not answer from your own knowledge.",
        expected_output="The weather report directly from the tool.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    return crew
