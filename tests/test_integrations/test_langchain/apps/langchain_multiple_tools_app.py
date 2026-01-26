"""
Multiple Tools LangChain App: LLM with multiple tools
Complexity: MEDIUM - Tests calling different tools based on query

Uses FakeMessagesListChatModel for deterministic tool calls.
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig, RunnableLambda


@tool
def get_weather(city: str) -> str:
    """Returns the current weather in a city."""
    weather_data = {
        "san francisco": "Foggy, 58F",
        "new york": "Sunny, 72F",
        "london": "Rainy, 55F",
        "tokyo": "Cloudy, 68F",
        "paris": "Partly cloudy, 62F",
    }
    return weather_data.get(
        city.lower(), f"Weather data not available for {city}"
    )


@tool
def get_population(city: str) -> str:
    """Returns the population of a city."""
    population_data = {
        "san francisco": "874,000",
        "new york": "8,336,000",
        "london": "8,982,000",
        "tokyo": "13,960,000",
        "paris": "2,161,000",
    }
    return population_data.get(
        city.lower(), f"Population data not available for {city}"
    )


@tool
def get_timezone(city: str) -> str:
    """Returns the timezone of a city."""
    timezone_data = {
        "san francisco": "PST (UTC-8)",
        "new york": "EST (UTC-5)",
        "london": "GMT (UTC+0)",
        "tokyo": "JST (UTC+9)",
        "paris": "CET (UTC+1)",
    }
    return timezone_data.get(
        city.lower(), f"Timezone data not available for {city}"
    )


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result."""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"


tools = [get_weather, get_population, get_timezone, calculate]
tools_by_name = {t.name: t for t in tools}


def create_multiple_tools_chain(tool_calls_config: list, final_message: str):
    """Create a chain that executes multiple tool calls."""

    def run_chain(inputs: dict, config: RunnableConfig = None):
        messages = inputs.get("messages", [])

        # Create responses for this invocation (no bind_tools needed)
        responses = [
            AIMessage(content="", tool_calls=tool_calls_config),
            AIMessage(content=final_message),
        ]
        llm = FakeMessagesListChatModel(responses=responses)

        # First LLM call
        response = llm.invoke(messages, config=config)
        messages_with_response = list(messages) + [response]

        # Execute tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if tool_name in tools_by_name:
                    result = tools_by_name[tool_name].invoke(
                        tool_args, config=config
                    )
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

            # Second LLM call
            final_response = llm.invoke(messages_with_response, config=config)
            return {"messages": messages_with_response + [final_response]}

        return {"messages": messages_with_response}

    return RunnableLambda(run_chain)


# Pre-configured chains for different test scenarios
city_info_tool_calls = [
    {
        "name": "get_weather",
        "args": {"city": "Tokyo"},
        "id": "call_001",
        "type": "tool_call",
    },
    {
        "name": "get_population",
        "args": {"city": "Tokyo"},
        "id": "call_002",
        "type": "tool_call",
    },
    {
        "name": "get_timezone",
        "args": {"city": "Tokyo"},
        "id": "call_003",
        "type": "tool_call",
    },
]

mixed_tool_calls = [
    {
        "name": "get_weather",
        "args": {"city": "Paris"},
        "id": "call_001",
        "type": "tool_call",
    },
    {
        "name": "calculate",
        "args": {"expression": "100 * 1.5 + 50"},
        "id": "call_002",
        "type": "tool_call",
    },
]

city_info_chain = create_multiple_tools_chain(
    city_info_tool_calls,
    "Here is the information about Tokyo: Weather is Cloudy at 68F, Population is 13,960,000, and Timezone is JST (UTC+9).",
)
mixed_chain = create_multiple_tools_chain(
    mixed_tool_calls,
    "Paris weather is partly cloudy at 62F. The calculation 100 * 1.5 + 50 = 200.0.",
)


def invoke_city_info(inputs: dict, config: RunnableConfig = None):
    """Invoke chain that gets city info (weather, population, timezone)."""
    return city_info_chain.invoke(inputs, config=config)


def invoke_mixed_tools(inputs: dict, config: RunnableConfig = None):
    """Invoke chain that uses weather and calculate tools."""
    return mixed_chain.invoke(inputs, config=config)
