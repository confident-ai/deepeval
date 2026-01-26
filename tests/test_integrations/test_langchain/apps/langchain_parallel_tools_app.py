"""
Parallel Tools LangChain App: LLM that calls multiple tools in parallel
Complexity: HIGH - Tests parallel tool execution

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


tools = [get_weather, get_stock_price, get_exchange_rate, calculate]
tools_by_name = {t.name: t for t in tools}


def create_parallel_chain(tool_calls_config: list, final_message: str):
    """Create a chain that executes tools in parallel."""

    def run_chain(inputs: dict, config: RunnableConfig = None):
        messages = inputs.get("messages", [])

        responses = [
            AIMessage(content="", tool_calls=tool_calls_config),
            AIMessage(content=final_message),
        ]
        llm = FakeMessagesListChatModel(responses=responses)

        # First LLM call - gets all parallel tool calls
        response = llm.invoke(messages, config=config)
        messages_with_response = list(messages) + [response]

        # Execute all tool calls
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

            # Final LLM call with all results
            final_response = llm.invoke(messages_with_response, config=config)
            return {"messages": messages_with_response + [final_response]}

        return {"messages": messages_with_response}

    return RunnableLambda(run_chain)


# Pre-configured parallel tool call scenarios

# Parallel weather queries for multiple cities
parallel_weather_calls = [
    {
        "name": "get_weather",
        "args": {"city": "Tokyo"},
        "id": "call_w1",
        "type": "tool_call",
    },
    {
        "name": "get_weather",
        "args": {"city": "New York"},
        "id": "call_w2",
        "type": "tool_call",
    },
    {
        "name": "get_weather",
        "args": {"city": "London"},
        "id": "call_w3",
        "type": "tool_call",
    },
]

# Mixed parallel tools
parallel_mixed_calls = [
    {
        "name": "get_weather",
        "args": {"city": "Paris"},
        "id": "call_m1",
        "type": "tool_call",
    },
    {
        "name": "get_stock_price",
        "args": {"symbol": "TSLA"},
        "id": "call_m2",
        "type": "tool_call",
    },
    {
        "name": "get_exchange_rate",
        "args": {"from_currency": "USD", "to_currency": "EUR"},
        "id": "call_m3",
        "type": "tool_call",
    },
    {
        "name": "calculate",
        "args": {"expression": "100 * 1.5"},
        "id": "call_m4",
        "type": "tool_call",
    },
]

# Parallel stock queries
parallel_stock_calls = [
    {
        "name": "get_stock_price",
        "args": {"symbol": "AAPL"},
        "id": "call_s1",
        "type": "tool_call",
    },
    {
        "name": "get_stock_price",
        "args": {"symbol": "GOOGL"},
        "id": "call_s2",
        "type": "tool_call",
    },
    {
        "name": "get_stock_price",
        "args": {"symbol": "MSFT"},
        "id": "call_s3",
        "type": "tool_call",
    },
    {
        "name": "get_stock_price",
        "args": {"symbol": "TSLA"},
        "id": "call_s4",
        "type": "tool_call",
    },
    {
        "name": "get_stock_price",
        "args": {"symbol": "AMZN"},
        "id": "call_s5",
        "type": "tool_call",
    },
]

weather_chain = create_parallel_chain(
    parallel_weather_calls,
    "Weather report: Tokyo is Sunny at 72F, New York is Cloudy at 58F, London is Rainy at 52F.",
)
mixed_chain = create_parallel_chain(
    parallel_mixed_calls,
    "Here are all the results: Paris weather is partly cloudy at 65F, TSLA is $245.60, USD/EUR is 0.92, and 100*1.5 = 150.",
)
stocks_chain = create_parallel_chain(
    parallel_stock_calls,
    "Stock prices: AAPL $178.50, GOOGL $142.30, MSFT $378.90, TSLA $245.60, AMZN $185.20.",
)


def invoke_parallel_weather(inputs: dict, config: RunnableConfig = None):
    """Invoke parallel weather queries for multiple cities."""
    return weather_chain.invoke(inputs, config=config)


def invoke_parallel_mixed(inputs: dict, config: RunnableConfig = None):
    """Invoke parallel mixed tools."""
    return mixed_chain.invoke(inputs, config=config)


def invoke_parallel_stocks(inputs: dict, config: RunnableConfig = None):
    """Invoke parallel stock price queries."""
    return stocks_chain.invoke(inputs, config=config)
