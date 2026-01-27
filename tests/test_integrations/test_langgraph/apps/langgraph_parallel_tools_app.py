"""
Parallel Tool Execution LangGraph Agent
Complexity: HIGH - Tests parallel tool execution and aggregation
"""

from typing import Literal

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    weather = {
        "tokyo": "Sunny, 72°F",
        "new york": "Cloudy, 58°F",
        "london": "Rainy, 52°F",
        "paris": "Partly cloudy, 65°F",
        "sydney": "Clear, 78°F",
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
def search_news(topic: str) -> str:
    """Search for news about a topic."""
    news = {
        "tech": "Tech stocks rally as AI boom continues",
        "finance": "Federal Reserve signals rate stability",
        "weather": "Climate change accelerating, report finds",
        "sports": "World Cup preparations underway",
    }
    for key, value in news.items():
        if key in topic.lower():
            return value
    return f"No news found for {topic}"


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return f"{expression} = {eval(expression)}"
        return "Invalid expression"
    except:
        return "Calculation error"


tools = [
    get_weather,
    get_stock_price,
    get_exchange_rate,
    search_news,
    calculate,
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)


def agent_node(state: dict) -> dict:
    """Agent that can call multiple tools in parallel."""
    messages = state["messages"]
    system_prompt = HumanMessage(
        content="""You are a helpful assistant with access to multiple tools.
        When asked for multiple pieces of information, call all relevant tools in parallel.
        For example, if asked about weather in multiple cities, call get_weather for each city."""
    )
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


async def async_agent_node(state: dict) -> dict:
    """Async agent that can call multiple tools in parallel."""
    messages = state["messages"]
    system_prompt = HumanMessage(
        content="""You are a helpful assistant with access to multiple tools.
        When asked for multiple pieces of information, call all relevant tools in parallel."""
    )
    response = await llm_with_tools.ainvoke([system_prompt] + messages)
    return {"messages": [response]}


def should_continue(state: dict) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def build_sync_app():
    """Build sync app for parallel tool execution."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def build_async_app():
    """Build async app for parallel tool execution."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", async_agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


sync_app = build_sync_app()
async_app = build_async_app()
