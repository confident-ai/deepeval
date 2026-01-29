"""
tests/test_integrations/test_openai_agents/apps/tool_chain_app.py
Tests sequential tool dependencies (Tool A -> Output -> Tool B).
"""

from deepeval.openai_agents import Agent, function_tool

@function_tool
def get_lat_long(city: str) -> str:
    """Get the latitude and longitude for a specific city."""
    # Deterministic mock
    if "tokyo" in city.lower():
        return "35.6762,139.6503"
    elif "paris" in city.lower():
        return "48.8566,2.3522"
    return "0,0"

@function_tool
def get_weather_at_coords(lat_long: str) -> str:
    """Get the weather for specific coordinates (format: 'lat,long')."""
    # This tool relies strictly on the output of the previous tool
    if "35.6762,139.6503" in lat_long:
        return "Clear, 22°C"
    elif "48.8566,2.3522" in lat_long:
        return "Cloudy, 18°C"
    return "Unknown Weather"

def get_tool_chain_app():
    """
    Returns an agent that MUST chain tools to answer.
    """
    agent = Agent(
        name="Sequential Tool Agent",
        instructions=(
            "You are a weather assistant. "
            "To get the weather, you MUST first get the coordinates using 'get_lat_long', "
            "and THEN use 'get_weather_at_coords' with those exact coordinates. "
            "Do not guess the coordinates."
        ),
        tools=[get_lat_long, get_weather_at_coords],
        tool_use_behavior="run_llm_again",
    )
    
    return agent, "What is the weather in Tokyo?"