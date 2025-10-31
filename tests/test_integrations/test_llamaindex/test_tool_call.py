import os
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from tests.test_integrations.utils import generate_trace_json, assert_trace_json


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    # Mock weather data for testing
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 65°F",
        "Tokyo": "Rainy, 68°F",
        "Paris": "Partly cloudy, 70°F",
    }
    return weather_data.get(
        location, f"Weather data not available for {location}"
    )


tool = FunctionTool.from_defaults(
    get_weather,
    # async_fn=aget_weather,  # optional!
)

llm = OpenAI(model="gpt-4o-mini")
agent = ReActAgent(llm=llm, tools=[tool])

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "tool_call.json")


async def run_agent():
    await agent.run("what is the weather in sf")


# @generate_trace_json(json_path)
@assert_trace_json(json_path)
async def test_execute_agent():
    await run_agent()
