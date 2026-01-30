"""
Tool OpenAI Agent
Complexity: MEDIUM - Uses DeepEval's function_tool wrapper
"""

from agents import Agent, ModelSettings
from deepeval.openai_agents import function_tool


# Use DeepEval's wrapper to test tool tracking
@function_tool
def get_weather(city: str) -> str:
    """Returns the current weather in a city."""
    # Deterministic mock data
    weather_data = {
        "san francisco": "Foggy, 58°F",
        "new york": "Sunny, 72°F",
        "london": "Rainy, 55°F",
        "tokyo": "Cloudy, 68°F",
    }
    return weather_data.get(
        city.lower(), f"Weather data not available for {city}"
    )


@function_tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        # Safe deterministic eval
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return f"{expression} = {eval(expression)}"
        return "Invalid expression"
    except Exception:
        return "Error"


agent = Agent(
    name="ToolAgent",
    instructions="You are a helper. Use tools for weather or math. Do not answer from memory.",
    model="gpt-4o",
    tools=[get_weather, calculate],
    model_settings=ModelSettings(temperature=0.0),
)
