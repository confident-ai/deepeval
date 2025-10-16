import os
import json
import asyncio
import pytest
from tests.test_integrations.utils import assert_trace_json, generate_trace_json

from crewai import Task, Agent, LLM, Crew
from crewai.tools import tool

# from deepeval.integrations.crewai import Crew, Agent, LLM
from deepeval.integrations.crewai import instrument_crewai
from deepeval.tracing import trace


@tool
def get_weather(city: str) -> str:
    """Fetch weather data for a given city. Returns temperature and conditions."""
    weather_data = {
        "New York": {
            "temperature": "72°F",
            "condition": "Partly Cloudy",
            "humidity": "65%",
        },
        "London": {
            "temperature": "60°F",
            "condition": "Rainy",
            "humidity": "80%",
        },
        "Tokyo": {
            "temperature": "75°F",
            "condition": "Sunny",
            "humidity": "55%",
        },
        "Paris": {
            "temperature": "68°F",
            "condition": "Cloudy",
            "humidity": "70%",
        },
        "Sydney": {
            "temperature": "82°F",
            "condition": "Clear",
            "humidity": "50%",
        },
    }

    if city in weather_data:
        weather = weather_data[city]
        return f"Weather in {city}: {weather['temperature']}, {weather['condition']}, Humidity: {weather['humidity']}"
    else:
        return f"Weather in {city}: 70°F, Clear, Humidity: 60% (default data)"


agent = Agent(
    role="Weather Reporter",
    goal="Provide accurate and helpful weather information to users.",
    backstory="An experienced meteorologist who loves helping people plan their day with accurate weather reports.",
    tools=[get_weather],
    verbose=True,
)

task = Task(
    description="Get the current weather for {city} and provide a helpful summary.",
    expected_output="A clear weather report including temperature, conditions, and humidity.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
)

################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "crewai.json")


# @generate_trace_json(json_path)
@assert_trace_json(json_path)
def test_crewai():
    crew.kickoff({"city": "London"})


if __name__ == "__main__":
    instrument_crewai()
    test_crewai()
