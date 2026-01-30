import os
import json
import asyncio
import pytest
from tests.test_integrations.utils import assert_trace_json, generate_trace_json
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.tracing.context import current_trace_context, current_span_context

from crewai import Task
from crewai.tools import tool

from deepeval.integrations.crewai import Crew, Agent, LLM, tool
from deepeval.integrations.crewai import instrument_crewai, reset_crewai_instrumentation
from deepeval.tracing import trace


@tool(metric_collection="test_collection_1")
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


################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "crewai_component.json")


# @generate_trace_json(json_path)
@assert_trace_json(json_path)
def test_crewai_component():
    reset_crewai_instrumentation()
    trace_manager.clear_traces()
    test_exporter.clear_span_json_list()
    trace_testing_manager.test_dict = None
    
    current_trace_context.set(None)
    current_span_context.set(None)
    # Initialize inside test to ensure fresh state
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0,
        metric_collection="test_collection_1",
    )

    agent = Agent(
        role="Weather Reporter",
        goal="Provide accurate and helpful weather information to users.",
        backstory="An experienced meteorologist who loves helping people plan their day with accurate weather reports.",
        tools=[get_weather],
        verbose=True,
        llm=llm,
        metric_collection="test_collection_1",
    )

    task = Task(
        description="Get the current weather for {city} and provide a helpful summary.",
        expected_output="A clear weather report including temperature, conditions, and humidity.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        metric_collection="test_collection_1",
    )

    with trace(
        name="crewai",
        tags=["crewai"],
        metadata={"crewai": "crewai"},
        user_id="crewai",
        thread_id="crewai",
        metric_collection="test_collection_1",
    ):
        crew.kickoff({"city": "London"})


if __name__ == "__main__":
    instrument_crewai()
    test_crewai_component()
