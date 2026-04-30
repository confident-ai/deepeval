"""
Multiple Tools LangChain App: LLM with multiple tools
Complexity: MEDIUM - Tests calling different tools based on query

Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
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


# City info tools
city_info_tools = [get_weather, get_population, get_timezone]
city_info_tools_by_name = {t.name: t for t in city_info_tools}

# Mixed tools
mixed_tools = [get_weather, calculate]
mixed_tools_by_name = {t.name: t for t in mixed_tools}

# LLMs with tool bindings
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)
llm_city_info = llm.bind_tools(city_info_tools)
llm_mixed = llm.bind_tools(mixed_tools)


def _run_multi_tool_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Run a multi-tool chain."""
    messages = inputs.get("messages", [])

    response = llm_with_tools.invoke(messages, config=config)
    messages_with_response = list(messages) + [response]

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in tools_by_name:
                # Use full tool_call structure to trigger proper callbacks
                tool_call_input = {
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_id,
                    "type": "tool_call",
                }
                result = tools_by_name[tool_name].invoke(
                    tool_call_input, config=config
                )
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

        final_response = llm_with_tools.invoke(
            messages_with_response, config=config
        )
        return {"messages": messages_with_response + [final_response]}

    return {"messages": messages_with_response}


async def _arun_multi_tool_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Async run a multi-tool chain."""
    messages = inputs.get("messages", [])

    response = await llm_with_tools.ainvoke(messages, config=config)
    messages_with_response = list(messages) + [response]

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in tools_by_name:
                # Use full tool_call structure to trigger proper callbacks
                tool_call_input = {
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_id,
                    "type": "tool_call",
                }
                result = await tools_by_name[tool_name].ainvoke(
                    tool_call_input, config=config
                )
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

        final_response = await llm_with_tools.ainvoke(
            messages_with_response, config=config
        )
        return {"messages": messages_with_response + [final_response]}

    return {"messages": messages_with_response}


# Create wrapper functions that will be wrapped in RunnableLambda
def _city_info_sync(inputs: dict, config: RunnableConfig = None):
    return _run_multi_tool_chain(
        inputs, llm_city_info, city_info_tools_by_name, config=config
    )


async def _city_info_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_multi_tool_chain(
        inputs, llm_city_info, city_info_tools_by_name, config=config
    )


def _mixed_tools_sync(inputs: dict, config: RunnableConfig = None):
    return _run_multi_tool_chain(
        inputs, llm_mixed, mixed_tools_by_name, config=config
    )


async def _mixed_tools_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_multi_tool_chain(
        inputs, llm_mixed, mixed_tools_by_name, config=config
    )


# Wrap as RunnableLambda chains for proper callback event propagation
_city_info_chain = RunnableLambda(_city_info_sync).with_config(
    run_name="city_info_chain"
)
_city_info_async_chain = RunnableLambda(_city_info_async).with_config(
    run_name="city_info_chain"
)
_mixed_tools_chain = RunnableLambda(_mixed_tools_sync).with_config(
    run_name="mixed_tools_chain"
)
_mixed_tools_async_chain = RunnableLambda(_mixed_tools_async).with_config(
    run_name="mixed_tools_chain"
)


def invoke_city_info(inputs: dict, config: RunnableConfig = None):
    """Invoke chain that gets city info (weather, population, timezone)."""
    return _city_info_chain.invoke(inputs, config=config)


async def ainvoke_city_info(inputs: dict, config: RunnableConfig = None):
    """Async invoke chain that gets city info."""
    return await _city_info_async_chain.ainvoke(inputs, config=config)


def invoke_mixed_tools(inputs: dict, config: RunnableConfig = None):
    """Invoke chain that uses weather and calculate tools."""
    return _mixed_tools_chain.invoke(inputs, config=config)


async def ainvoke_mixed_tools(inputs: dict, config: RunnableConfig = None):
    """Async invoke chain that uses weather and calculate tools."""
    return await _mixed_tools_async_chain.ainvoke(inputs, config=config)
