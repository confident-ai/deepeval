import random

from crewai import Task, Crew, Agent
from crewai.tools import tool

from deepeval.integrations.crewai import instrument_crewai
from tests.test_integrations.utils import assert_trace_json, generate_trace_json
import os

# instrument_crewai()


@tool
def get_weather(city: str) -> str:
    """Fetch weather data for a given city. Returns temperature and conditions."""
    weather_data = {
        "New York": "Partly Cloudy",
        "London": "Rainy",
        "Tokyo": "Sunny",
        "Paris": "Cloudy",
        "Sydney": "Clear",
    }

    condition = weather_data.get(city, "Clear")
    temperature = f"{random.randint(45, 95)}°F"
    humidity = f"{random.randint(30, 90)}%"

    return (
        f"Weather in {city}: {temperature}, {condition}, Humidity: {humidity}"
    )


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
    tracing=False
)

_current_dir = os.path.dirname(os.path.abspath(__file__))


# @generate_trace_json(
#     json_path=os.path.join(_current_dir, "test_sync_eval.json"),
#     is_run=True
# )
@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_sync_eval.json"), is_run=True
)
def test_run_sync_eval():
    from deepeval.tracing import trace
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.dataset import EvaluationDataset, Golden

    answer_relavancy_metric = AnswerRelevancyMetric()

    dataset = EvaluationDataset(
        goldens=[
            Golden(input="London"),
            Golden(input="Paris"),
        ]
    )

    for golden in dataset.evals_iterator():
        with trace(metrics=[answer_relavancy_metric]):
            crew.kickoff({"city": golden.input})
