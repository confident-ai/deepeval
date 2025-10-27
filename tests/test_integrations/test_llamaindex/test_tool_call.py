import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index import instrument_llama_index

instrument_llama_index(instrument.get_dispatcher())

def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    # Mock weather data for testing
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 65°F", 
        "Tokyo": "Rainy, 68°F",
        "Paris": "Partly cloudy, 70°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


tool = FunctionTool.from_defaults(
    get_weather,
    # async_fn=aget_weather,  # optional!
)

llm = OpenAI(model="gpt-4o-mini")
agent = ReActAgent(llm=llm, tools=[tool])

async def llm_app(input: str):
    return await agent.run(input)

asyncio.run(llm_app("what is the weather in sf"))