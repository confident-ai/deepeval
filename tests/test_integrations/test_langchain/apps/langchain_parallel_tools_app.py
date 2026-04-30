"""
Parallel Tools LangChain App: LLM that calls multiple tools in parallel
Complexity: HIGH - Tests parallel tool execution

Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig, RunnableLambda


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    weather = {
        "tokyo": "Sunny, 72F",
        "new york": "Cloudy, 58F",
        "london": "Rainy, 52F",
        "paris": "Partly cloudy, 65F",
        "sydney": "Clear, 78F",
    }
    return weather.get(city.lower(), f"No weather data for {city}")


@tool
def get_stock_price(symbol: str) -> str:
    """Get stock price for a symbol."""
    prices = {
        "AAPL": "$178.50",
        "GOOGL": "$142.30",
        "MSFT": "$378.90",
        "TSLA": "$245.60",
        "AMZN": "$185.20",
    }
    return prices.get(symbol.upper(), f"No price for {symbol}")


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get exchange rate between currencies."""
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("EUR", "USD"): 1.09,
    }
    key = (from_currency.upper(), to_currency.upper())
    if key in rates:
        return f"1 {from_currency.upper()} = {rates[key]} {to_currency.upper()}"
    return f"No rate for {from_currency} to {to_currency}"


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return f"{expression} = {eval(expression)}"
        return "Invalid expression"
    except Exception:
        return "Calculation error"


# Weather-only tools
weather_tools = [get_weather]
weather_tools_by_name = {t.name: t for t in weather_tools}

# Mixed parallel tools
mixed_tools = [get_weather, get_stock_price, get_exchange_rate, calculate]
mixed_tools_by_name = {t.name: t for t in mixed_tools}

# Stock-only tools
stock_tools = [get_stock_price]
stock_tools_by_name = {t.name: t for t in stock_tools}

# LLMs with parallel tool calling enabled
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)
llm_weather = llm.bind_tools(weather_tools, parallel_tool_calls=True)
llm_mixed = llm.bind_tools(mixed_tools, parallel_tool_calls=True)
llm_stocks = llm.bind_tools(stock_tools, parallel_tool_calls=True)


def _run_parallel_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Run a parallel tool chain."""
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


async def _arun_parallel_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Async run a parallel tool chain."""
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


# Create wrapper functions for RunnableLambda
def _parallel_weather_sync(inputs: dict, config: RunnableConfig = None):
    return _run_parallel_chain(
        inputs, llm_weather, weather_tools_by_name, config=config
    )


async def _parallel_weather_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_parallel_chain(
        inputs, llm_weather, weather_tools_by_name, config=config
    )


def _parallel_mixed_sync(inputs: dict, config: RunnableConfig = None):
    return _run_parallel_chain(
        inputs, llm_mixed, mixed_tools_by_name, config=config
    )


async def _parallel_mixed_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_parallel_chain(
        inputs, llm_mixed, mixed_tools_by_name, config=config
    )


def _parallel_stocks_sync(inputs: dict, config: RunnableConfig = None):
    return _run_parallel_chain(
        inputs, llm_stocks, stock_tools_by_name, config=config
    )


async def _parallel_stocks_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_parallel_chain(
        inputs, llm_stocks, stock_tools_by_name, config=config
    )


# Wrap as RunnableLambda chains for proper callback event propagation
_parallel_weather_chain = RunnableLambda(_parallel_weather_sync).with_config(
    run_name="parallel_weather_chain"
)
_parallel_weather_async_chain = RunnableLambda(
    _parallel_weather_async
).with_config(run_name="parallel_weather_chain")
_parallel_mixed_chain = RunnableLambda(_parallel_mixed_sync).with_config(
    run_name="parallel_mixed_chain"
)
_parallel_mixed_async_chain = RunnableLambda(_parallel_mixed_async).with_config(
    run_name="parallel_mixed_chain"
)
_parallel_stocks_chain = RunnableLambda(_parallel_stocks_sync).with_config(
    run_name="parallel_stocks_chain"
)
_parallel_stocks_async_chain = RunnableLambda(
    _parallel_stocks_async
).with_config(run_name="parallel_stocks_chain")


# Weather functions
def invoke_parallel_weather(inputs: dict, config: RunnableConfig = None):
    """Invoke parallel weather queries for multiple cities."""
    return _parallel_weather_chain.invoke(inputs, config=config)


async def ainvoke_parallel_weather(inputs: dict, config: RunnableConfig = None):
    """Async invoke parallel weather queries."""
    return await _parallel_weather_async_chain.ainvoke(inputs, config=config)


# Mixed functions
def invoke_parallel_mixed(inputs: dict, config: RunnableConfig = None):
    """Invoke parallel mixed tools."""
    return _parallel_mixed_chain.invoke(inputs, config=config)


async def ainvoke_parallel_mixed(inputs: dict, config: RunnableConfig = None):
    """Async invoke parallel mixed tools."""
    return await _parallel_mixed_async_chain.ainvoke(inputs, config=config)


# Stock functions
def invoke_parallel_stocks(inputs: dict, config: RunnableConfig = None):
    """Invoke parallel stock price queries."""
    return _parallel_stocks_chain.invoke(inputs, config=config)


async def ainvoke_parallel_stocks(inputs: dict, config: RunnableConfig = None):
    """Async invoke parallel stock price queries."""
    return await _parallel_stocks_async_chain.ainvoke(inputs, config=config)
