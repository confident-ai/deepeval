"""
LangGraph Agent with Multiple Tools
Complexity: MEDIUM - Multiple tools, agent selects appropriate ones
"""

from typing import Literal

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def get_weather(city: str) -> str:
    """Returns the current weather in a city."""
    weather_data = {
        "san francisco": "Foggy, 58°F",
        "new york": "Sunny, 72°F",
        "london": "Rainy, 55°F",
        "tokyo": "Cloudy, 68°F",
        "paris": "Partly cloudy, 62°F",
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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: dict) -> dict:
    """Call the LLM with current messages."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: dict) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def build_app():
    """Build and compile the multi-tool agent graph."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


app = build_app()
