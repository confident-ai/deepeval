"""
Agent LlamaIndex App
Complexity: HIGH - Standard ReAct Agent with Function Tools
"""

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


def get_weather(city: str) -> str:
    """Useful for getting the weather for a specific city."""
    # Deterministic mock data
    weather_map = {
        "san francisco": "Foggy, 15C",
        "new york": "Sunny, 25C",
        "london": "Rainy, 10C",
        "tokyo": "Cloudy, 20C",
    }
    return weather_map.get(city.lower(), f"Weather data unknown for {city}")


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


def get_agent():
    """Builds and returns a ReAct agent with deterministic tools."""
    tools = [
        FunctionTool.from_defaults(fn=get_weather),
        FunctionTool.from_defaults(fn=multiply),
    ]

    # Deterministic LLM
    llm = OpenAI(model="gpt-4o", temperature=0.0)

    # Use constructor injection instead of .from_tools()
    # and strict system prompt to ensure tools are called
    return ReActAgent(
        tools=tools,
        llm=llm,
        verbose=True,
        system_prompt="You are a helpful assistant. You MUST use the provided tools to answer questions. Do not answer from your own knowledge.",
    )
