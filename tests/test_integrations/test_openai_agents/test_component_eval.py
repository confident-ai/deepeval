from agents import add_trace_processor
import asyncio
import random
import requests
from deepeval.openai_agents import DeepEvalTracingProcessor
from deepeval.openai_agents import Runner, Agent
from deepeval.openai_agents import function_tool

add_trace_processor(DeepEvalTracingProcessor())


@function_tool(metric_collection="test_collection_1")
def get_current_weather(latitude: float, longitude: float) -> dict:
    """
    Fetches weather data for a given location using the Open-Meteo API.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.

    Returns:
        dict: A dictionary containing the weather data or error message.
    """
    # Return random dummy weather data for testing purposes
    return {
        "temperature_2m": round(random.uniform(-10, 35), 1),
        "humidity": random.randint(20, 100),
        "apparent_temperature": round(random.uniform(-15, 40), 1),
        "precipitation": round(random.uniform(0, 20), 1),
        "weather_code": random.choice([0, 1, 2, 3, 45, 48, 51, 61, 71, 80, 95]),
        "wind_speed_10m": round(random.uniform(0, 30), 1),
        "wind_direction_10m": random.randint(0, 360),
        "dummy": True,
    }


@function_tool
def get_location_coordinates(city_name: str) -> dict:
    """
    Get latitude and longitude for a city name.

    Args:
        city_name (str): Name of the city

    Returns:
        dict: Dictionary with lat, lng coordinates
    """
    # Mock implementation - use real geocoding API in production
    locations = {
        "london": {"lat": 51.5074, "lng": -0.1278},
        "tokyo": {"lat": 35.6762, "lng": 139.6503},
        "new york": {"lat": 40.7128, "lng": -74.0060},
    }

    city_lower = city_name.lower()
    if city_lower in locations:
        return locations[city_lower]
    return {"error": f"Location not found: {city_name}"}


# Create the weather specialist agent
weather_agent = Agent(
    name="Weather Specialist Agent",
    instructions="""
    You are a weather agent. When providing current weather information 
    (including temperature, humidity, wind speed/direction, precipitation, and weather codes), provide:
    
    1. A clear and concise summary of the weather conditions.
    2. Practical suggestions or precautions for outdoor activities, health, or clothing based on the weather.
    3. If severe weather is detected (e.g., heavy rain, thunderstorms, extreme temperatures), 
       highlight necessary safety measures.
    
    Format your response in two sections:
    Weather Summary:
    - Briefly describe the weather in plain language.
    
    Suggestions:
    - Offer actionable advice relevant to the weather conditions.
    """,
    tools=[get_location_coordinates, get_current_weather],
    tool_use_behavior="run_llm_again",
    metric_collection="test_collection_1",
    # metrics=[task_completion_metric]
)


async def run_weather_agent(user_input: str):
    """Run the weather agent with user input"""
    runner = Runner()
    result = await runner.run(
        weather_agent, user_input, metric_collection="test_collection_1"
    )
    return result.final_output


# Usage example
async def main():
    user_query = "What's the weather like in London today?"
    response = await run_weather_agent(user_query)
    print(f"Agent Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
