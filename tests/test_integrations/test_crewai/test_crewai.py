import os
import json
import asyncio
import pytest
from tests.test_integrations.utils import (
    assert_json_object_structure,
    load_trace_data,
)
from deepeval.tracing.trace_test_manager import trace_testing_manager
from crewai import Task, Crew, Agent
from crewai.tools import tool


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


@pytest.mark.asyncio
async def test_json_schema():
    """
    Test the json schema of the trace. Raises an exception if the schema is invalid.
    """
    try:
        trace_testing_manager.test_name = json_path
        crew.kickoff({"city": "London"})
        actual_dict = await trace_testing_manager.wait_for_test_dict()
        expected_dict = load_trace_data(json_path)

        print(actual_dict)

        assert assert_json_object_structure(expected_dict, actual_dict)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None


################################ Generate Actual JSON Dump Code #################################


async def generate_actual_json_dump():
    try:
        trace_testing_manager.test_name = json_path
        crew.kickoff({"city": "London"})
        actual_dict = await trace_testing_manager.wait_for_test_dict()

        with open(json_path, "w") as f:
            json.dump(actual_dict, f)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None
