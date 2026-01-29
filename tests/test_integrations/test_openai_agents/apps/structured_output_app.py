"""
tests/test_integrations/test_openai_agents/apps/structured_output_app.py
Tests agent execution with enforced Structured Outputs (Pydantic).
"""

from pydantic import BaseModel
from deepeval.openai_agents import Agent


# Define the expected output structure
class WeatherReport(BaseModel):
    city: str
    temperature: float
    condition: str
    is_sunny: bool


def get_structured_output_app():
    """
    Returns an agent that MUST output a JSON object matching the WeatherReport model.
    """
    agent = Agent(
        name="Structured Weather Bot",
        instructions="You are a weather bot. Always report the weather for San Francisco as 20.5 degrees and Sunny.",
        # Feature: Structured Output
        output_type=WeatherReport,
    )

    return agent, "What is the weather in SF?"
